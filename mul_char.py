import cv2
import numpy as np
import os
from config import args


def charSeperate(src_img, filter_size=3):
    """函数功能：字符分割
       @param src_img: 输入图像
       @param filter_size: 中值滤波核大小
       @return dst_img: 分割出的字符图像列表"""

    # 灰度图
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 颜色反转（黑变白，白变黑），便于轮廓提取
    binary_inv = cv2.bitwise_not(binary)

    # 中值滤波
    binary_f = cv2.medianBlur(binary_inv, filter_size)

    # 查找字符区域
    # contours, hierarchy = cv2.findContours(image, mode, method)
    # mode（轮廓检索模式）：
    # cv2.RETR_EXTERNAL：只检测外轮廓。
    #cv2.CHAIN_APPROX_SIMPLE：只存储端点（压缩水平、垂直和对角线的点），更高效。

    contours, _ = cv2.findContours(binary_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #contours = [cnt1, cnt2, cnt3, ..., cntN]
    #cnt.shape == (n, 1, 2)  # n 是该轮廓的点数 1是 OpenCV 的设计格式（每个点作为一个“行向量”）2：表示每个点的坐标 [x, y]

    # 遍历所有区域，寻找最大宽度
    w_max = 0
    for cnt in contours:
        _, _, w, _ = cv2.boundingRect(cnt)  # （矩形左上角的横坐标，纵坐标，矩形的宽度，高度）
        if w > w_max:
            w_max = w

    # 根据轮廓的横向位置将字符区域合并在一起，避免一个字符被分成多个小轮廓
    # 遍历所有区域，拼接x坐标接近的区域
    char_dict = {}
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_mid = x + w // 2  # 计算中点位置

        # 如果char_dict是空的（第一个字符），直接添加。
        # 对于已有的所有字符区域中点z，是否都与当前中点x_mid的距离大于半个最大字符宽度（w_max // 2），说明为新字符。
        if not char_dict or all(np.abs(z - x_mid) > w_max // 2 for z in char_dict):
            char_dict[x_mid] = cnt
        else:
            # 否则，当前轮廓距离某个已有字符区域很近，说明它们可能是同一个字符被切成多个小块
            # 找到接近的字符区域（通过中点坐标比较），用 np.concatenate() 把它们合并成一个更大的轮廓。
            for z in list(char_dict.keys()):
                if np.abs(z - x_mid) <= w_max // 2:
                    char_dict[z] = np.concatenate((char_dict[z], cnt), axis=0)  # 拼接两个区域

    # 按照中点坐标，对字符进行排序
    char_dict = dict(sorted(char_dict.items(), key=lambda item: item[0]))

    # 遍历所有区域，提取字符
    dst_img = []
    for _, cnt in char_dict.items():
        x, y, w, h = cv2.boundingRect(cnt)
        roi = binary[y:y + h, x:x + w]
        dst_img.append(roi)

    return dst_img


if __name__ == "__main__":
    # 读取本地图像
    input_path = "./sentence_img/yyy.png"
    image = cv2.imread(input_path)

    if image is None:
        print(f"无法读取图像: {input_path}")
        exit()

    # 字符分割
    characters = charSeperate(image, filter_size=3)

    # 输出目录
    output_dir = args.result + "/output_chars"
    os.makedirs(output_dir, exist_ok=True)

    # 保存字符图像
    for idx, char_img in enumerate(characters):
        output_path = os.path.join(output_dir, f"char_{idx + 1}.png")
        cv2.imwrite(output_path, char_img)
        print(f"字符{idx + 1}保存至：{output_path}")
