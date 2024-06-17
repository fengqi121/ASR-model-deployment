package whisper.transform
//导入工具包
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean
//声音处理类
class AudioTensorSource {
    companion object {
        private const val bytesPerFloat = 4//浮点数占4个字节
        private const val sampleRate = 16000//采样率
        private const val maxAudioLengthInSeconds = 30//最大录音时间



        @SuppressLint("MissingPermission")
        fun fromRecording(stopRecordingFlag: AtomicBoolean): OnnxTensor {
            val recordingChunkLengthInSeconds = 1
            //获取最小缓冲区大小
            val minBufferSize = maxOf(
                AudioRecord.getMinBufferSize(
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_FLOAT
                ),
                2 * recordingChunkLengthInSeconds * sampleRate * bytesPerFloat
            )
            //创建一个音频记录器
            val audioRecord = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.MIC)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(sampleRate)
                        .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .build()
                )
                .setBufferSizeInBytes(minBufferSize)
                .build()

            try {
                //创建一个浮点数数组，用于存储音频数据
                val floatAudioData = FloatArray(maxAudioLengthInSeconds * sampleRate) { 0.0f }
                //浮点数数据偏移量
                var floatAudioDataOffset = 0
                //开始录音
                audioRecord.startRecording()

                while (!stopRecordingFlag.get() && floatAudioDataOffset < floatAudioData.size) {
                    //计算需要读取的浮点数
                    val numFloatsToRead = minOf(
                        recordingChunkLengthInSeconds * sampleRate,
                        floatAudioData.size - floatAudioDataOffset
                    )
                    //读取音频数据
                    val readResult = audioRecord.read(
                        floatAudioData, floatAudioDataOffset, numFloatsToRead,
                        AudioRecord.READ_BLOCKING
                    )
                    //打印日志
                    Log.d(MainActivity.TAG, "AudioRecord.read(float[], ...) returned $readResult")
                    //如果读取结果大于等于0
                    if (readResult >= 0) {
                        floatAudioDataOffset += readResult
                    } else {
                        throw RuntimeException("AudioRecord.read() returned error code $readResult")
                    }
                }
                //停止录音
                audioRecord.stop()
               //获取环境
                val env = OrtEnvironment.getEnvironment()
                val floatAudioDataBuffer = FloatBuffer.wrap(floatAudioData)
                //返回一个音频张量
                return OnnxTensor.createTensor(
                    env, floatAudioDataBuffer,
                    tensorShape(1, floatAudioData.size.toLong())
                )

            } finally {
                //释放音频记录器
                if (audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    audioRecord.stop()
                }
                audioRecord.release()
            }
        }
    }

}