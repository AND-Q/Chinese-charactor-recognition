# 手写汉字识别

## 项目简介

本项目是一个基于深度学习的中文字符识别系统，能够识别单个中文字符以及连续的中文文本。项目使用了多种卷积神经网络模型，如ResNet、VGG、LeNet等，通过对大量中文手写字符样本的学习，实现了对中文字符的高精度识别。

## 功能特点

- **单字符识别**：能够准确识别单个中文字符
- **多字符识别**：支持对连续中文文本的分割与识别
- **多模型支持**：实现了多种深度学习模型，包括ResNet、ResNet18、ResNet34、VGG、LeNet和CNN
- **数据增强**：通过多种图像处理技术进行数据增强，提高模型鲁棒性
- **可视化分析**：提供训练过程的损失函数和准确率可视化

## 项目结构

```
Chinese_charactor_recognition/
├── model.py               # 模型定义文件，包含多种CNN模型结构
├── train.py               # 模型训练代码
├── test.py                # 模型测试代码
├── recognize.py           # 字符识别实现
├── mul_char.py            # 多字符分割处理
├── MyDataset.py           # 数据集加载与预处理
├── config.py              # 配置参数
├── dict.py                # 字符字典处理
├── plt.py                 # 绘图工具
├── vgg_block.py           # VGG模型块定义
├── Residual_block.py      # ResNet残差块定义
├── look.py                # 数据可视化工具
├── data/                  # 数据集目录
│   ├── HWDB1.1trn_gnt     # 训练数据
│   ├── HWDB1.1tst_gnt     # 测试数据
│   └── char_dict          # 字符映射字典
├── result/                # 结果保存目录
│   ├── param/             # 模型参数
│   ├── fig/               # 训练过程图表
│   └── output_chars/      # 分割后的字符图像
└── sentence_img/          # 待识别的句子图像
```

## 技术实现

### 数据集

本项目使用CASIA-HWDB（中国科学院手写数据库）作为训练和测试数据集。数据集包含大量中文手写字符样本，以.gnt格式存储。项目中的`MyDataset.py`负责解析.gnt文件，提取字符图像和对应标签。

### 模型架构

项目实现了多种卷积神经网络模型：

- **ResNet**：使用残差连接的深层卷积网络，有效解决深层网络的梯度消失问题
- **VGG**：使用小卷积核堆叠的经典卷积网络
- **LeNet**：早期经典的卷积神经网络，结构简单但有效
- **CNN**：基础卷积神经网络
- **ResNet18/34**：更深层次的残差网络变体

### 数据增强

为提高模型泛化能力，项目实现了多种数据增强技术：

- 对比度和亮度调整
- 椒盐噪声和高斯噪声
- 高斯模糊和锐化
- 随机旋转
- 随机翻转和缩放

### 多字符识别

多字符识别通过以下步骤实现：

1. 图像预处理（灰度化、二值化、反色）
2. 中值滤波去噪
3. 轮廓检测提取字符区域
4. 合并相近轮廓，避免单个字符被过度分割
5. 对分割出的单个字符进行识别
6. 合并识别结果

## 使用方法

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-username/Chinese_charactor_recognition.git
cd Chinese_charactor_recognition

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py --model ResNet --epochs 50 --batch_size 512
```

可选参数：
- `--model`：选择模型类型（ResNet, VGG, LeNet, CNN, ResNet18, ResNet34）
- `--epochs`：训练轮数
- `--batch_size`：批次大小
- `--lr`：学习率

### 测试模型

```bash
python test.py --model ResNet
```

### 识别单个字符

```bash
python recognize.py
```

### 识别多字符文本

```bash
# 先分割字符
python mul_char.py

# 然后识别分割后的字符
python recognize.py
```

## 实验结果

在CASIA-HWDB数据集上，识别准确率80%以上。


## 未来改进

- 实现基于注意力机制的端到端多字符识别
- 添加对手写体和印刷体的混合识别支持
- 优化模型结构，减小模型大小，提高推理速度
- 增加对罕见字和生僻字的识别支持
- 实现基于Web的在线识别服务

## 参考资料

- CASIA-HWDB数据集：http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html
- ResNet论文：He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
- VGG论文：Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition.

## 许可证

MIT License 
