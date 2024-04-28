import onnxruntime as ort
import numpy as np

# 加载模型
sess = ort.InferenceSession('tiny.onnx')

# 创建输入数据
mels = np.random.randn(1,80,3000).astype(np.float32)
tokens = np.random.randint(0,51865,(1,448)).astype(np.int64)

# 进行推理
input_name_1 = sess.get_inputs()[0].name
input_name_2 = sess.get_inputs()[1].name
result = sess.run(None, {input_name_1: mels, input_name_2: tokens})

# 输出结果
print(result)