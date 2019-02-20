import os
import cv2
# EIGEN
import load_image

def cvt(imgpath):
    img = cv2.imread(imgpath)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    raws, cols = img_gray.shape
    for i in range(raws):
        for j in range(cols):
            if img_gray[i][j] != 0:
                img_gray[i][j] = 255
    save_name = imgpath.split(os.path.sep)[-1][:-4]
    cv2.imwrite('trimaps_with_window/{}.png'.format(save_name),img_gray)


if __name__ == '__main__':
    imglist = load_image.loadim('masks_with_window')
    for imgfile in imglist:
        cvt(imgfile)

