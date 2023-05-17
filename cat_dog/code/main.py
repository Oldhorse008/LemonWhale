import torch
import torch.nn as nn
import torchvision
import os
import time
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# 定义设备

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集


class mydata(Dataset):
    def __init__(self, path, transform=None):
        self.transforms = transform
        self.dog_path = os.path.join(path, "dog")
        self.cat_path = os.path.join(path, "cat")
        self.dog_list = os.listdir(self.dog_path)
        self.cat_list = os.listdir(self.cat_path)

    def __len__(self):
        return len(self.cat_list) + len(self.dog_list)

    def __getitem__(self, index):
        if index < len(self.dog_list):
            image_path = os.path.join(self.dog_path, self.dog_list[index])
            image = Image.open(image_path)
            label = 0
        else:
            image_path = os.path.join(
                self.cat_path, self.cat_list[index - len(self.dog_list)]
            )
            image = Image.open(image_path)
            label = 1
        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


## 定义transform
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

train_data_path = "dataset/train"
vali_data_path = "dataset/test"

train_dataset = mydata(train_data_path, transform)
vali_dataset = mydata(vali_data_path, transform)

print("训练集的长度为：{}".format(len(train_dataset)))
print("验证集的长度为：{}".format(len(vali_dataset)))

# 载入数据
Batch = 16

train_dataloader = DataLoader(train_dataset, batch_size=Batch, shuffle=True)
vali_dataloader = DataLoader(vali_dataset, batch_size=Batch, shuffle=True)

# 载入模型
model = torchvision.models.vgg16(pretrained=True)

num_fc = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_fc, 2)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier[6].parameters():
    param.requires_grad = True

model.to(device)

# 定义损失函数和优化器
LR = 0.05

loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

optim_fn = torch.optim.SGD(params=model.parameters(), lr=LR)

# 初始化summarywriter

writer = SummaryWriter("log")

# 训练
Epoch = 5

start_time = time.time()

for epoch in range(Epoch):
    print("<--------第{}轮训练开始-------->".format(epoch + 1))

    train_loss = 0
    train_acc = 0
    train_step = 1

    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = model(imgs)

        train_loss = loss_fn(outputs, targets)
        train_loss.backward()

        optim_fn.step()
        optim_fn.zero_grad()

        acc = (outputs.argmax(1) == targets).sum()
        train_acc += acc

        if train_step % 10 == 0:
            print("第{}轮第{}次的训练损失为：{}".format(epoch + 1, train_step, train_loss))
            print(
                "第{}轮第{}次的训练正确率为：{}".format(
                    epoch + 1, train_step, train_acc / (train_step * Batch)
                )
            )

        train_step += 1
    if (epoch + 1) % 5 == 0 or epoch == Epoch:
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc / len(train_dataset), epoch)

        torch.save(model.state_dict(), "model/model_{}.pth".format(epoch + 1))
        print("第{}轮训练模型保存成功".format(epoch + 1))

end_time = time.time()

# 测试
loss = 0
acc = 0
with torch.no_grad():
    for data in vali_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)

        outputs = model(imgs)
        vali_loss = loss_fn(outputs, targets)
        loss += vali_loss

        vali_acc = (outputs.argmax(1) == targets).sum()
        acc += vali_acc
        print(acc.item())
        print(len(vali_dataset))

print("验证集上的损失为：{}".format(loss.item()))
print("验证集上的准确率为：{}".format(acc.item() / len(vali_dataset)))
print("训练用时：{:.2f}s".format(end_time - start_time))
