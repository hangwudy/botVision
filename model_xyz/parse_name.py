import os
import re


def loadim(image_path, ext='png'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext):
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path, filename)
            image_list.append(image_abs_path)
    return image_list


if __name__ == '__main__':
    imList = loadim('/home/hangwu/Repositories/AIBox/AttitudeNet/xyz/render/cropped')
    nameList = []
    xList = []
    yList = []
    zList = []
    for i in imList:
        name = i.split(os.path.sep)[-1]
        match = re.match(r'([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(_+)([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(_+)([A-Za-z_]+)(_+)([\-|0-9][0-9]*)(\.png)', name, re.I)

        print(match.groups())
        xList.append(match.groups()[2])
        yList.append(match.groups()[6])
        zList.append(match.groups()[10])
        nameList.append(name)
    print(nameList)
    print(xList)
    print(yList)
    print(zList)
