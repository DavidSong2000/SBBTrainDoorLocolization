from PIL import Image
from single_img_bounding import single_img_bounding


img = Image.open('../test/IMG_3145_JPG.rf.cdbe2a4f1d6e46cabccd88c33326ab59.jpg').convert('RGB')


processed_img = single_img_bounding(img)


processed_img.save('output.png')
