import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


path = "test/cat.png"

origin_image = Image.open(path).convert("RGB")

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.CenterCrop((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

image = transform(origin_image).to("cuda")
image = torch.reshape(image, (-1, 3, 224, 224))

print(image.shape)

model = torchvision.models.vgg16()
num_fc = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_fc, 2)
model.load_state_dict(torch.load("model/model_10.pth"))
model.to("cuda")

model.eval()
with torch.no_grad():
    output = model(image).to("cuda")

print(output)

# 使用softmax函数对结果进行归一化，计算出概率最大的值  dim=1表示第一维度
output = F.softmax(output, dim=1)
# 将output的值复制到cpu上并转化成numpy格式
output = output.data.cpu().numpy()
print(output)
list = {0: "dog", 1: "cat"}
# 找出output中的最大值的索引
a = int(output.argmax(1))
print(output[0, a])

plt.figure()
plt.suptitle("识别结果为:{}   概率为：{:.1%}".format(list[a], output[0, a]))
plt.imshow(origin_image)
plt.show()
