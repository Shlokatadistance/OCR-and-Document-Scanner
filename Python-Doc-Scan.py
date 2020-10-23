from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import boto3
from PIL import ImageEnhance
from PIL import Image

def blur(image):
    b = cv2.Laplacian(image, cv2.CV_64F).var()
    return b

def transform(image):
    ratio = image.shape[0] / 500
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    d=image.shape
    h = image.shape[0]
    w = image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
                screenCnt = approx
                break
    if len(approx) == 4:              
        warped,a= transformation(orig, screenCnt.reshape(4, 2) * ratio)
        cv2.imwrite(r'C:\Users\AdityaJoshi\Desktop\zapfin\enhance.jpg',warped)
        if(len(approx)==4 and a<40000):
                image=crop(image)
                b=bright(image)
                if b < 150:
                    enhance(1.2)
                else:
                    enhance(1.4)
                exit();
        return len(approx)

def transformation(image, pts):
rect = order_points(pts)
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))
dst = np.array([
[0, 0],
[maxWidth - 1, 0],
[maxWidth - 1, maxHeight - 1],
[0, maxHeight - 1]], dtype = "float32")
        
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
return warped,maxWidth*maxHeight

def order_points(pts):
rect = np.zeros((4, 2), dtype = "float32")
s = pts.sum(axis = 1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]
diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]
return rect

def crop(image):
    orig = image.copy()
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask=mask)
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))
    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, 2)
    contours, hierarchy = cv2.findContours(img_th,
                                           cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    max_brightness = 0
    canvas = image.copy()
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w*h > 40000:
            mask = np.zeros(image.shape, np.uint8)
            mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness
            break

    if w*h > 40000:
        x, y, w, h = brightest_rectangle
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imwrite(r'',mask)
        img = cv2.imread(r'')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        crop = img[y:y+h,x:x+w]
        d = crop.shape
        h = crop.shape[0]
        w = crop.shape[1]
        r1=h/w
        r2=w/h
        if r1 < 0.26 or r2 < 0.26:
            cv2.imwrite(r'',image)
            b=bright(orig)
            if b < 150:
                enhance(1.2)
            else:
                enhance(1.4)
            exit();
        cv2.imwrite(r'',crop)
        e=Image.open(r'')
        return e

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def bright(image):
    b=np.mean(image)
    return b

def enhance(x):
    e=Image.open(r'')
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
    image_sharped.save(r'')
    enhanced=cv2.imread(r'')
    cv2.imshow('enhanced',imutils.resize(enhanced, height = 500))

image = cv2.imread(r'')
cv2.imshow('image',imutils.resize(image, height = 500))
b=blur(image)
if b>90:
        c=transform(image)
        if c==4:
            b=bright(image)
            if b < 150:
                enhance(1.2)
            else:
                enhance(1.4)
        else:
            image=crop(image)
            b=bright(image)
            if b < 150:
                enhance(1.2)
            else:
                enhance(1.4)     
else:
    print("Blur Image.Click again.")