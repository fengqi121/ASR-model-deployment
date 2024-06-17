package whisper.transform
//导入必要包
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.extensions.OrtxPackage
import android.os.SystemClock
//声音识别类
class SpeechRecognizer(modelBytes: ByteArray) : AutoCloseable {
    private val session: OrtSession//会话
    private val baseInputs: Map<String, OnnxTensor>//基础输入
//初始化
    init {
        val env = OrtEnvironment.getEnvironment()//获取环境
        val sessionOptions = OrtSession.SessionOptions()//会话选项
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())//注册自定义操作库

        session = env.createSession(modelBytes, sessionOptions)//创建会话
        //基础输入
        baseInputs = mapOf(
            "min_length" to createIntTensor(env, intArrayOf(1), tensorShape(1)),
            "max_length" to createIntTensor(env, intArrayOf(200), tensorShape(1)),
            "num_beams" to createIntTensor(env, intArrayOf(1), tensorShape(1)),
            "num_return_sequences" to createIntTensor(env, intArrayOf(1), tensorShape(1)),
            "length_penalty" to createFloatTensor(env, floatArrayOf(1.0f), tensorShape(1)),
            "repetition_penalty" to createFloatTensor(env, floatArrayOf(1.0f), tensorShape(1)),
        )
    }
//结果类
    data class Result(val text: String, val inferenceTimeInMs: Long)
//运行类
    fun run(audioTensor: OnnxTensor): Result {
        //输入
        val inputs = mutableMapOf<String, OnnxTensor>()
        baseInputs.toMap(inputs)
        inputs["audio_pcm"] = audioTensor
        val startTimeInMs = SystemClock.elapsedRealtime()//开始时间
        val outputs = session.run(inputs)//运行
        val elapsedTimeInMs = SystemClock.elapsedRealtime() - startTimeInMs//花费时间
        val recognizedText = outputs.use {
            @Suppress("UNCHECKED_CAST")
            (outputs[0].value as Array<Array<String>>)[0][0]
        }//识别文本
        return Result(recognizedText, elapsedTimeInMs)//返回结果
    }
//关闭onnxtensor对象
    override fun close() {
        baseInputs.values.forEach {
            it.close()
        }
        session.close()
    }
}