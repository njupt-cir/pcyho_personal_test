# coding:utf-8

import numpy as np
import cv2

img = np.mat(np.ones((300, 300), dtype=np.uint8))
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 颜色设置——灰度图
print(img.shape)
cv2.imshow('test', img)  # 显示图片
cv2.waitKey(0)  # 冻结屏幕？
