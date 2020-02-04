import cv2
import pytesseract
from PIL import Image
#import numpy as np
import os
print("hello World")
path = r'path from your system'
#path = r'/Users/singularity/Documents/ocrsample.png'
img = cv2.imread(path,0)

cv2.imshow('image',img)
f = open("sampleocr.txt","w+")

#x = "{}.png".format(os.getpid())
#cv2.imwrite('x', img)
text = pytesseract.image_to_string(img)
f.write(text)
f.close()

#os.remove(filename)
print(text)
path2 = r''
img2 =cv2.imread(path,0)
gray = cv2.cvtColor(img2,cv2,COLOR_BGR2GRAY)
canny = cv2.Canny(gray,100,500)
#x represents my image here, below is an image enhancement code
def enhance(x):
    e=Image.open(r'path')
    enh_bri = ImageEnhance.Brightness(e)
    brightness = 1
    image_brightened = enh_bri.enhance(brightness)
    enh_col = ImageEnhance.Color(image_brightened)
    color = 1.2
    image_colored = enh_col.enhance(color)
    enh_con = ImageEnhance.Contrast(image_colored)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    enh_sha = ImageEnhance.Sharpness(image_contrasted)
    sharpness = 2
    image_sharped = enh_sha.enhance(sharpness)
    image_sharped.save(r'path')
    enhanced=cv2.imread(r'path')
    cv2.imshow('enhanced',imutils.resize(enhanced, height = 500))
cv2.waitKey()
cv2.destroyAllWindows()
