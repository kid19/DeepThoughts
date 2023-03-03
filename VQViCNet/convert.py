from PIL import Image
import os
import string
from matplotlib import pyplot as plt

path = "E:\\pythonProject\\Datasets\\COVID-19_Lung_CT_Scans_datasets\\Non-COVID\\"  # 最后要加双斜杠，不然会报错
filelist = os.listdir(path)

for file in filelist:
    whole_path = os.path.join(path, file)
    img = Image.open(whole_path)  # 打开图片img = Image.open(dir)#打开图片
    img = img.convert("RGB")  # 将一个4通道转化为rgb三通道
    save_path = 'E:\\pythonProject\\Datasets\\COVID-19_Lung_CT_Scans_datasets\\RGB_NonCOVID\\'
    img.save(save_path + file)
