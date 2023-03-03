# # 批量修改尺寸
# import os
# from PIL import Image
#
# dir_img = "./data/covid_data2/non-COVID/"
# dir_save = "./data/covid_data2/new_non_covid/"
# size = (224, 224)
#
# # 获取目录下所有图片名
# list_temp = os.listdir(dir_img)
# list_img = list_temp[1:]  # 因为列表中第0项为Mac OS X操作系统所创造的隐藏文件  .DS_Store，所以从第一项开始取
#
# # 获得路径、打开要修改的图片
# for img_name in list_img:
#     img_path = dir_img + img_name
#     old_image = Image.open(img_path)
#     save_path = dir_save + img_name
#
#     # 保存修改尺寸后的图片
#     old_image.resize(size, Image.ANTIALIAS).save(save_path)
# print("Done!")