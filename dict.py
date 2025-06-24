import os
import pickle
import struct
import numpy as np
from config import args

train_data_dir = os.path.join(args.root, 'HWDB1.1trn_gnt')
test_data_dir = os.path.join(args.root, 'HWDB1.1tst_gnt')

# 获取字典
char_set = set()
data_path = test_data_dir
#从.gnt文件中逐样本读取图像和对应标签的二进制数据，并解析为灰度图像和字符编码。
for file_name in os.listdir(data_path):
    if file_name.endswith('.gnt'):
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'rb') as f:
            header_size = 10
            while True:
                header = np.fromfile(f, dtype='uint8', count=header_size)
                if not header.size:
                    break
                sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
                tagcode = header[5] + (header[4] << 8)
                width = header[6] + (header[7] << 8)
                height = header[8] + (header[9] << 8)

                if header_size + width * height != sample_size:
                    break
                image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                #将 tagcode 16位编码 按大端方式打包成2字节，再按 gb2312 解码为对应汉字。
                #加入到char_set中
                tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
                char_set.add(tagcode_unicode)

char_list = sorted(list(char_set))  # set是随机的，需要排一下序
#创建字符→索引的字典
char_dict = dict(zip(char_list, range(len(char_list))))
#使用pickle序列化保存该字典到本地
f = open(args.root + '/char_dict', 'wb')
pickle.dump(char_dict, f)  # 将码表写入到数据集
f.close()

for char, idx in char_dict.items():
    print(f"{char}: {idx}")
