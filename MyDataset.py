import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
import struct
import pickle
from config import args
import os


class MyDataset(Dataset):
    def __init__(self, data_path, transforms, augment=False):
        super(MyDataset, self).__init__()
        self.images = []
        self.labels = []
        self.transforms = transforms
        self.augment = augment  # 是否启用数据增强
        self.get_data(data_path)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        # 数据增强
        if self.augment:
            image = self.apply_augmentations(image)

        # 应用基础变换
        image = self.transforms(image)#toTenser
        return image, label

    def __len__(self):
        return len(self.labels)

    def get_data(self, data_path):  # 读取并解析.gnt文件
        with open(args.root + '/char_dict', 'rb') as f:
            char_dict = pickle.load(f)

        for file_name in os.listdir(data_path):
            if file_name.endswith('.gnt'):
                file_path = os.path.join(data_path, file_name)
                with open(file_path, 'rb') as f:
                    # .gnt文件的每个样本头占用10个字节。
                    header_size = 10

                    while True:
                        header = np.fromfile(f, dtype='uint8', count=header_size)
                        if not header.size: break
                        # 解析样本字节数（低位在前，高位在后)
                        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
                        # 解析汉字标签（使用GB2312编码转为字符），并转为分类编号。
                        tag_code = header[5] + (header[4] << 8)
                        tag_code = struct.pack('>H', tag_code).decode('gb2312')
                        label = char_dict[tag_code]#编号
                        # 获取图像宽高
                        width = header[6] + (header[7] << 8)
                        height = header[8] + (header[9] << 8)
                        # 确保样本大小合理
                        if header_size + width * height != sample_size: break
                        # 读取图像数据
                        image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                        # 将图像二值化并反色（黑底白字 → 白底黑字)
                        _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
                        # 统一图像尺寸为模型输入需要的大小
                        image = cv2.resize(image, (args.image_size, args.image_size))
                        # 存入图像与对应标签。
                        self.images.append(image)
                        self.labels.append(label)

    def apply_augmentations(self, image):
        """应用随机数据增强"""
        # 转换为PIL Image进行增强处理
        pil_img = Image.fromarray(image)

        # 颜色抖动（在二值化图像上调整对比度和亮度）
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(random.uniform(0.8, 1.2))

        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(random.uniform(0.8, 1.2))

        # 转换为numpy数组进行处理
        image = np.array(pil_img)

        # 椒盐噪声
        if random.random() < 0.1:
            noise_type = random.choice(['gaussian', 'saltpepper'])
            if noise_type == 'saltpepper':
                prob = 0.02
                output = np.copy(image)
                r = np.random.rand(*image.shape)
                output[r < prob / 2] = 0
                output[r > 1 - prob / 2] = 255
                image = output
            else:  # 高斯噪声（需临时转换为浮点型）
                gauss = np.random.normal(0, random.uniform(0, 15), image.shape)
                noisy = image.astype(np.float32) + gauss
                image = np.clip(noisy, 0, 255).astype(np.uint8)

        # 高斯模糊，锐化
        if random.random() < 0.5:
            if random.random() < 0.5:
                image = cv2.GaussianBlur(image, (3, 3), 0)
            else:
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
                image = cv2.filter2D(image, -1, kernel)

        # 几何变换
        # 随机旋转（-5到5度）
        if random.random() < 0.7:
            angle = random.uniform(-5, 5)
            h, w = image.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)

        # 随机翻转
        # if random.random() < 0.5:
        #     flip_code = random.choice([-1, 0, 1])  # 0:垂直, 1:水平, -1:双向
        #     image = cv2.flip(image, flip_code)

        # # 随机缩放（90%-110%）
        # if random.random() < 0.3:
        #     scale = random.uniform(0.9, 1.1)
        #     h, w = image.shape
        #     new_w = int(w * scale)
        #     new_h = int(h * scale)
        #     image = cv2.resize(image, (new_w, new_h))
        #
        #     # 保持原始尺寸
        #     if scale < 1.0:  # 填充
        #         delta_w = w - new_w
        #         delta_h = h - new_h
        #         top, bottom = delta_h // 2, delta_h - delta_h // 2
        #         left, right = delta_w // 2, delta_w - delta_w // 2
        #         image = cv2.copyMakeBorder(image, top, bottom, left, right,
        #                                    cv2.BORDER_REPLICATE)
        #     else:  # 裁剪
        #         start_x = (new_w - w) // 2
        #         start_y = (new_h - h) // 2
        #         image = image[start_y:start_y + h, start_x:start_x + w]

        return image
