import matplotlib.pyplot as plt

# 从你的训练日志中提取数据（这里需要替换为实际数据）
epochs = range(1, 51)
train_losses = [
    8.1319, 7.3019, 4.7110, 2.1364, 1.1908, 0.8361, 0.6494, 0.5300, 0.4437, 0.3762,
    0.3242, 0.2728, 0.2329, 0.1976, 0.1658, 0.2595, 0.1202, 0.0961, 0.0802, 0.0684,
    0.0584, 0.0515, 0.0443, 0.0383, 0.0349, 0.0316, 0.0286, 0.0257, 0.0228, 0.0212,
    0.0191, 0.0181, 0.0158, 0.0154, 0.0146, 0.0128, 0.0134, 0.0108, 0.0122, 0.0099,
    0.0107, 0.0089, 0.0092, 0.0100, 0.0090, 0.0075, 0.0060, 0.0077, 0.0070, 0.0071
]

train_accuracies = [
    0.0012, 0.0077, 0.1385, 0.4973, 0.6975, 0.7814, 0.8276, 0.8573, 0.8793, 0.8968,
    0.9103, 0.9236, 0.9339, 0.9435, 0.9519, 0.9355, 0.9642, 0.9710, 0.9753, 0.9784,
    0.9811, 0.9833, 0.9854, 0.9874, 0.9886, 0.9896, 0.9906, 0.9914, 0.9926, 0.9932,
    0.9939, 0.9942, 0.9950, 0.9951, 0.9954, 0.9960, 0.9957, 0.9967, 0.9960, 0.9969,
    0.9966, 0.9973, 0.9971, 0.9969, 0.9972, 0.9978, 0.9982, 0.9976, 0.9978, 0.9979
]

test_accuracies = [
    0.0027, 0.0203, 0.2841, 0.5575, 0.6981, 0.7571, 0.7673, 0.8040, 0.8077, 0.8237,
    0.8240, 0.8375, 0.8274, 0.8402, 0.8362, 0.8405, 0.8409, 0.8364, 0.8104, 0.8435,
    0.8258, 0.8452, 0.8379, 0.8491, 0.8497, 0.8487, 0.8493, 0.8497, 0.8511, 0.8466,
    0.8534, 0.8466, 0.8554, 0.8484, 0.8532, 0.8540, 0.8572, 0.8551, 0.8580, 0.8595,
    0.8578, 0.8602, 0.8599, 0.8565, 0.8612, 0.8568, 0.8610, 0.8604, 0.8577, 0.8514
]
# 设置全局绘图参数
plt.rcParams.update({
    'font.size': 12,        # 全局字体大小
    'figure.figsize': (10, 6),  # 图像尺寸
    'lines.linewidth': 2,   # 线条宽度
    'axes.grid': True,      # 显示网格
    'grid.linestyle': '--', # 网格线型
    'savefig.dpi': 300,     # 保存分辨率
    'savefig.bbox': 'tight' # 保存时自动裁剪
})

# 绘制Loss曲线
plt.figure()
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.close()

# 绘制Accuracy曲线
plt.figure()
plt.plot(epochs, train_accuracies, 'g-', label='Train Accuracy')
plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')

# 添加训练准确率最后一个点的标注
plt.scatter(epochs[-1], train_accuracies[-1], color='green')
plt.text(epochs[-1] + 0.5, train_accuracies[-1],
         f'{train_accuracies[-1]:.2%}', color='green', va='center')

# 添加测试准确率最后一个点的标注
plt.scatter(epochs[-1], test_accuracies[-1], color='red')
plt.text(epochs[-1] + 0.5, test_accuracies[-1],
         f'{test_accuracies[-1]:.2%}', color='red', va='center')

plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')  # 将图例放在右下角
plt.ylim(0, 1)  # 固定Y轴范围
plt.savefig('accuracy_curves.png')
plt.close()

print("图像已保存为 training_loss.png 和 accuracy_curves.png")