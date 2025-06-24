import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from config import args
from MyDataset import MyDataset

# 设置保存路径
save_dir = args.result+"/look_after"
os.makedirs(save_dir, exist_ok=True)

# 定义 transform（和训练中保持一致）
transform = transforms.Compose([
    transforms.ToTensor()
])

# 创建数据集对象
dataset = MyDataset(args.root + '/HWDB1.1trn_gnt', transforms=transform, augment=True)
# train_set = MyDataset(args.root + '/HWDB1.1trn_gnt', transforms=transform,augment=True)

# 保存前N个样本图像
num_samples = 10
for i in range(num_samples):
    image_tensor, label = dataset[i]

    # 保存为图像文件
    image_pil = transforms.ToPILImage()(image_tensor)
    image_pil.save(os.path.join(save_dir, f"sample_{i}_label_{label}.png"))

print(f"前 {num_samples} 张图像已保存到 {save_dir}")
