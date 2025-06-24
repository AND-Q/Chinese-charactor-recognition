import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from MyDataset import MyDataset
from config import args
import time
from model import LeNet, VGG, ResNet,CNN,ResNet18,ResNet34
from test import test as ts

# 获取训练集
transform = transforms.ToTensor()


train_set = MyDataset(args.root + '/HWDB1.1trn_gnt', transforms=transform,augment=False)
test_set = MyDataset(args.root + '/HWDB1.1tst_gnt', transforms=transform)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=16)

# 定义model 、loss 、optimizer
torch.manual_seed(args.seed)  # 设置随机数种子，确保结果可重复
device = torch.device('cuda' if args.cuda else 'cpu')

model_dict = {
    'LeNet': LeNet,
    'VGG': VGG,
    'ResNet': ResNet,
    'CNN':CNN,
    'ResNet18':ResNet18,
    'ResNet34':ResNet34
}

model = model_dict[args.model]().to(device)

if args.load_state:
    model.load_state_dict(torch.load(args.result + '/param/model.pth', map_location=device))  # 继续训练

model.train()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20, 50], gamma=0.1)

start_time = time.time()
train_losses, train_accuracies, test_accuracies = [], [], []
# 开始训练
for epoch in range(1, args.epochs + 1):
    running_loss = 0.0
    running_acc = 0.0

    for data in train_loader:
        image, label = data
        image, label = image.to(device), label.to(device)

        # 前向传播
        out = model(image)
        loss = loss_fn(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.item()

        # 反向传播
        optimizer.zero_grad()  # 梯度清零，以免影响其他batch
        loss.backward()  # 后向传播，计算梯度
        optimizer.step()  # 利用梯度更新W、b的参数
        # scheduler.step()  # 动态调整学习率

    # 每5个epoch保存一遍模型权重
    if epoch % 5 == 0:
        torch.save(model.state_dict(), args.result + '/param/model.pth')

    test_acc, f1, recall = ts(model)

    train_loss = running_loss / (len(train_set))
    train_acc = running_acc / (len(train_set))

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # 打印一个循环后，训练集上的loss和正确率
    print('Train {} epoch, Loss: {:.4f}, Train_Acc: {:.4f}, Test_Acc:{:.4f}'
          .format(epoch, train_loss, train_acc, test_acc))

end_time = time.time()
total_time = end_time - start_time
# 转换为时分秒
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)
print(f"\n训练总耗时：{hours:02d}:{minutes:02d}:{seconds:02d}（时:分:秒）")

# 画图
epochs = range(1, args.epochs + 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training & Test Metrics')
plt.legend()
plt.grid(True)
plt.savefig(args.result + "/fig/" + args.model)
plt.close()
