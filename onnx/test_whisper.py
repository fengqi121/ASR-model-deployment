import torch
import whisper
model=whisper.load_model(r"D:\tiny.pt")
# 创建输入数据
model.eval()
mels = torch.randn(1,80,3000)
tokens =torch.randint(0,51865,(1,448))
output=model(mels,tokens)
torch.onnx.export(model,(mels,tokens),'tiny.onnx',verbose=True)