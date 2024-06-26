package whisper.transform
//导入所需要的包
import android.widget.ImageButton
import android.Manifest
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean


class MainActivity : AppCompatActivity() {
//应用程序的主要组件
    private val recordAudioButton: ImageButton by lazy { findViewById(R.id.audio_button) }
    private val resultText: TextView by lazy { findViewById(R.id.result_text) }
    private val statusText: TextView by lazy { findViewById(R.id.status_text) }
    private val speechRecognizer: SpeechRecognizer by lazy {
        resources.openRawResource(R.raw.whisper_cpu_int8_model).use {
            val modelBytes = it.readBytes()
            SpeechRecognizer(modelBytes)
        }
    }

    private var recording = false//录音标志
    private val stopRecordingFlag = AtomicBoolean(false)//停止录音标志
    //定义线程执行器
    private val workerThreadExecutor = Executors.newSingleThreadExecutor()
    //输出结果
    private fun setSuccessfulResult(result: SpeechRecognizer.Result) {
        runOnUiThread {
            statusText.text = "语音识别花费(${result.inferenceTimeInMs} ms)。"
            resultText.text = result.text.ifEmpty { "<无语音需识别。>" }
        }
    }
    //处理异常
    private fun setError(exception: Exception) {
        Log.e(TAG, "Error: ${exception.localizedMessage}", exception)
        runOnUiThread {
            statusText.text = "Error"
            resultText.text = exception.localizedMessage
        }
    }
    //检查是否有录音权限
    private fun hasRecordAudioPermission(): Boolean =
        ActivityCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    //请求权限
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == RECORD_AUDIO_PERMISSION_REQUEST_CODE) {
            if (!hasRecordAudioPermission()) {
                Toast.makeText(
                    this,
                    "请打开录音权限",
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }


    //初始化
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

//录音的点击事件
        recordAudioButton.setOnClickListener {
            //检查是否有录音权限
            if (!hasRecordAudioPermission()) {
                requestPermissions(
                    arrayOf(Manifest.permission.RECORD_AUDIO),
                    RECORD_AUDIO_PERMISSION_REQUEST_CODE
                )
                return@setOnClickListener
            }
            recording=!recording//录音标志
            stopRecordingFlag.set(!recording)
            workerThreadExecutor.submit {
                try {
                    if (recording) {
                        runOnUiThread {
                            recordAudioButton.setImageResource(R.drawable.micphone_open)
                        }//录音状态
                    } else {
                        runOnUiThread {
                            recordAudioButton.setImageResource(R.drawable.micphone_closed)
                        }//默认状态
                    }
                    runOnUiThread {
                        recordAudioButton.isEnabled = true
                    }//录音按钮可用
                    if(recording) {
                            val audioTensor = AudioTensorSource.fromRecording(stopRecordingFlag)
                            val result = audioTensor.use { speechRecognizer.run(audioTensor) }
                            setSuccessfulResult(result)
                        }
                } catch (e: Exception) {
                    setError(e)
                }
            }

        }
    }
    //后台挂起时
    override fun onPause() {
        super.onPause()
        stopRecordingFlag.set(true)
    }
    //应用退出
    override fun onDestroy() {
        super.onDestroy()
        workerThreadExecutor.shutdown()
        speechRecognizer.close()
    }

    companion object {
        const val TAG = "ORTSpeechRecognizer"//标识i当前模块
        private const val RECORD_AUDIO_PERMISSION_REQUEST_CODE = 1
    }
}