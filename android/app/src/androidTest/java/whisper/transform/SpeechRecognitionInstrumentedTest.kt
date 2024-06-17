package whisper.transform


import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import junit.framework.TestCase.assertEquals
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import java.lang.reflect.Field


/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class SpeechRecognitionInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("whisper.transform", appContext.packageName)
    }
    fun testRecordingValueAfterButtonClick() {
        // 启动 MainActivity
        val scenario = ActivityScenario.launch(MainActivity::class.java)

        // 获取 MainActivity 的 recording 字段的初始值
        var recordingInitialValue = false
        scenario.onActivity { activity ->
            val field: Field = MainActivity::class.java.getDeclaredField("recording")
            field.isAccessible = true
            recordingInitialValue = field.get(activity) as Boolean
        }

        // 找到录音按钮并点击一次
        onView(withId(R.id.audio_button)).perform(click())

        // 获取 MainActivity 的 recording 字段的值
        var recordingValueAfterClick = false
        scenario.onActivity { activity ->
            val field: Field = MainActivity::class.java.getDeclaredField("recording")
            field.isAccessible = true
            recordingValueAfterClick = field.get(activity) as Boolean
        }

        // 验证 recording 字段的值是否已经改变
        Assert.assertNotEquals(recordingInitialValue, recordingValueAfterClick)
    }
}
