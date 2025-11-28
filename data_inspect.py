import numpy as np

# 加载文件
data = np.load('centerline/output/seg_0000/data/feature_normalized_distance.npy')

# 查看基本信息
print("形状 (Shape):", data.shape)
print("数据类型 (Dtype):", data.dtype)
print("最大值:", data.max(), "最小值:", data.min())

# 如果想看具体数值
print(data)