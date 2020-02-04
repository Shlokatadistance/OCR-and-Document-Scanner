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


cv2.waitKey()
cv2.destroyAllWindows()
