import numpy as np
import PIL.Image

import cv2

label_png = '123.png'
# lbl = np.asarray(PIL.Image.open(label_png))
# print(lbl.dtype)
# np.unique(lbl)
# print(lbl.shape)

img = cv2.imread(label_png, -1)
# rows, cols = img.shape
# for i in range(rows):
#     for j in range(cols):
#         if img[i][j] == 2:
#             # img[i][j] = 1
#             print(img[i][j])
i = 241
j = 129
print(img[i][j])
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_gray.shape)

# cv2.imshow('gray',img_gray)
# cv2.imshow('gray',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
