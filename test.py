import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from config import args
from MyDataset import MyDataset
from model import LeNet, VGG, ResNet,ResNet18,CNN,ResNet34
from sklearn.metrics import f1_score, recall_score

def test(model):
    # 获取测试集数据
    transform = transforms.ToTensor()
    test_set = MyDataset(args.root + '/HWDB1.1tst_gnt', transforms=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    device = torch.device('cuda' if args.cuda else 'cpu')
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0.0
    num_corrects = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            image, label = data
            image, label = image.to(device), label.to(device)

            out = model(image)
            loss = loss_fn(out, label)
            test_loss += loss.item() * label.size(0)

            _, pred = torch.max(out, dim=1)
            num_corrects += (pred == label).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # 计算F1和Recall
    f1 = f1_score(all_labels, all_preds, average='macro')  # 你也可以用 'weighted' 或 'micro'
    recall = recall_score(all_labels, all_preds, average='macro')

    avg_loss = test_loss / len(test_set)
    test_acc = num_corrects / len(test_set)
    # print('Test Loss: {:.4f}, Test Acc: {:.4f}'.format(avg_loss, test_acc))
    return test_acc, f1, recall

if __name__ == '__main__':
    # 加载Train模型
    device = torch.device('cuda' if args.cuda else 'cpu')
    model_dict = {
        'LeNet': LeNet,
        'VGG': VGG,
        'ResNet': ResNet,
        'ResNet18':ResNet18,
        'ResNet34':ResNet34,
        'CNN':CNN
    }
    model = model_dict[args.model]().to(device)
    model.load_state_dict(torch.load(args.result + '/param/model.pth', map_location=device))
    test_acc, f1, recall = test(model)
    print('Test Acc: {:.4f} F1 Score: {:.4f}, Recall: {:.4f}'.format(test_acc,f1, recall))