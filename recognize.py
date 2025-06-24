import pickle
import cv2
import torch
import os
from torch import nn
from torchvision import transforms
from model import LeNet, VGG, ResNet
from config import args

# 加载字符识别模型
device = torch.device('cuda' if args.cuda == 'true' else 'cpu')
model_dict = {
    'LeNet': LeNet,
    'VGG': VGG,
    'ResNet': ResNet
}

model = model_dict[args.model]().to(device)
model.load_state_dict(torch.load(args.result + '/param/model.pth', map_location=device))
model.eval()
loss_fn = nn.CrossEntropyLoss()

# 加载字符映射字典
with open(args.root + '/char_dict', 'rb') as f:
    char_dict = pickle.load(f)
    char_dict = {v: k for k, v in char_dict.items()}  # 转换为{编号: 字符}的格式


def process_image(image_path):
    """处理单张图片并返回预测结果"""
    # 读取图片并进行预处理
    image = cv2.imread(image_path, 0)
    if image is None:
        return None

    # 二值化和反色处理
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # 提取有效区域
    x, y, w, h = cv2.boundingRect(image)
    image = image[y:y + h, x:x + w]

    # 调整尺寸并转换为Tensor
    image = cv2.resize(image, (args.image_size, args.image_size))
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测结果
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
    return char_dict[int(pred[0])]


def recognize_chars(folder_path):
    #识别文件夹中图片的所有字符并拼接结果
    files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(x.split('_')[1].split('.')[0])  # 按char_后面的数字排序
    )

    # 逐个处理文件
    result = []
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        char = process_image(file_path)
        if char:
            result.append(char)
            print(f"识别文件 {filename} -> 字符: {char}")

    return ''.join(result)


if __name__ == "__main__":
    # 配置参数
    input_folder = args.result+"/output_chars"
    output_string = recognize_chars(input_folder)
    print("\n最终识别结果:", output_string)