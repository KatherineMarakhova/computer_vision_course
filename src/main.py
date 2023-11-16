import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

path = "bear.png"

img = cv2.imread(path)
cv2.imshow('Original Bear', img)
cv2.waitKey()

# 1 ORB FEATURES
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
keypoints = orb.detect(img,None)
kp, des = orb.compute(img, keypoints)
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv2.imshow('1 ORB FEATURES', img2)
cv2.waitKey()

# 2 SIFT FEATURES
img = cv2.imread(path)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp = sift.detect(gray,None)
img = cv2.drawKeypoints(gray,kp,img)
cv2.imshow('2 SIFT FEATURES', img)
cv2.waitKey()

# 3 CANNY EDGES
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img,100,200)
cv2.imshow('3 CANNY EDGES', edges)
cv2.waitKey()

# 4 GRAYSCALE
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
cv2.imshow('4 GRAYSCALE Bear', img)
cv2.waitKey()

# 5 PNG TO HSV
img = cv2.imread(path)
hsvImage = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
cv2.imshow('5 HSV image', hsvImage)
cv2.waitKey()

# 6 MIRROR THE IMAGE ON THE RIGHT EDGE
img = cv2.imread(path)
mirImage = cv2.flip(img, 1)
cv2.imshow('6 MIRROR THE IMAGE ON THE RIGHT EDGE', mirImage)
cv2.waitKey()

# 7 MIRROR THE IMAGE ON THE BOTTOM EDGE
img = cv2.imread(path)
mirImage = cv2.flip(img, 0)
cv2.imshow('7 MIRROR THE IMAGE ON THE BOTTOM EDGE', mirImage)
cv2.waitKey()

# 8 ROTATE THE IMAGE 45 DEGREES
img = cv2.imread(path)
(h, w, d) = img.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow('8 ROTATE THE IMAGE 45 DEGREES', rotated)
cv2.waitKey()

# 9 ROTATE THE IMAGE 30 DEGREES AROUND THE SPECIFIED POINT
img = cv2.imread(path)
(h, w, d) = img.shape
point = (w - 10, 0)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow('9 ROTATE THE IMAGE 30 DEGREES AROUND THE SPECIFIED POINT', rotated)
cv2.waitKey()

# 10 MOVE THE IMAGE 10 PIXELS TO THE RIGHT
img = cv2.imread(path)
h, w = img.shape[:2]
translation_matrix = np.float32([[1, 0, 10], [0, 1, 0]])
dst = cv2.warpAffine(img, translation_matrix, (w, h))
cv2.imshow('10 MOVE THE IMAGE 10 PIXELS TO THE RIGHT', dst)
cv2.waitKey(0)

# 11 CHANGE THE BRIGHTBESS
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

img = cv2.imread(path)
br_img = increase_brightness(img, value=50)
cv2.imshow('11 CHANGE THE BRIGHTBESS', br_img)
cv2.waitKey()

# 12 CHANGE THE CONTRAST
clahe = cv2.createCLAHE(clipLimit=-20, tileGridSize=(8,8))
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
l, a, b = cv2.split(lab)  # split on 3 different channels
l2 = clahe.apply(l)  # apply CLAHE to the L-channel
lab = cv2.merge((l2,a,b))  # merge channels
img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
cv2.imshow('12 CHANGE THE CONTRAST', img2)
cv2.waitKey()

# 13 GAMMA CORRECTION
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)
img = cv2.imread(path)
gammaImg = gammaCorrection(img, 2.2)
cv2.imshow('13 GAMMA CORRECTED IMAGE', gammaImg)
cv2.waitKey(0)

# 14 HISTOGRAM EQUALIZE
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
cv2.imshow('14 HISTOGRAM EQUALIZED', equalized)
cv2.waitKey()

#15 MAKING WARM
def making_warm(img):
    increase_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 155, 255])(range(256))
    decrease_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 45, 95, 255])(range(256))
    blue_ch, green_ch, red_ch = cv2.split(img)
    red_ch = cv2.LUT(red_ch, increase_table).astype(np.uint8)
    blue_ch = cv2.LUT(blue_ch, decrease_table).astype(np.uint8)
    output_image = cv2.merge((blue_ch, green_ch, red_ch))
    return output_image

img = cv2.imread(path)
warm = making_warm(img)
cv2.imshow('15 WARM IMG', warm)
cv2.waitKey()

#16 MAKING COLD
def making_cold(img):
    increase_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 155, 255])(range(256))
    decrease_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 45, 95, 255])(range(256))
    blue_ch, green_ch, red_ch = cv2.split(img)
    red_ch = cv2.LUT(red_ch, decrease_table).astype(np.uint8)
    blue_ch = cv2.LUT(blue_ch, increase_table).astype(np.uint8)
    output_image = cv2.merge((blue_ch, green_ch, red_ch))
    return output_image

img = cv2.imread(path)
warm = making_cold(img)
cv2.imshow('16 COLD IMG', warm)
cv2.waitKey()

#17 CHANGING PALETTE
def changing_palette(img, r, g, b):
    blue_ch, green_ch, red_ch = cv2.split(img)
    blue_ch+=b
    green_ch+=g
    red_ch+=r
    output_image = cv2.merge((blue_ch, green_ch, red_ch))
    return output_image
img = cv2.imread(path)
warm = changing_palette(img, 30 , 10, 10)
cv2.imshow('17 CHANGING PALETTE', warm)
cv2.waitKey()

# 18 IMAGE BINARIZATION
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
ret,thresh = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
cv2.imshow("18 IMAGE BINARIZATION", thresh )
cv2.waitKey()

#19 IMAGE BINARIZATION AND CONTOURS
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours( img, contours, -1, (0,0,256), 3, cv2.LINE_AA, hierarchy, 1 )
cv2.imshow('19 IMAGE BINARIZATION AND CONTOURS', img)
cv2.waitKey()

# 20 SOBEL'S CONVOLUTION
img = cv2.imread(path)
kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
im = cv2.filter2D(img, -1, kernel)
cv2.imshow("20 SOBEL'S CONVOLUTION", im )
cv2.waitKey()

# 21 BLUR
img = cv2.imread(path)
average_image = cv2.blur(img, ksize=(20, 20))
cv2.imshow('21 BLUR', average_image)
cv2.waitKey()

#22 LOW FOURIER FILTER
def highFourierFilter(image, d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(fshift))

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                if dis <= d:
                    transfor_matrix[i,j] = 1
                else:
                    transfor_matrix[i,j] = 0
        return transfor_matrix
    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

img = cv2.imread(path, 0)
new_img = highFourierFilter(img,60)
cv2.imshow('22 LOW FOURIER TRANSFORM', new_img)
cv2.waitKey()

# 23 HIGH FOURIER FILTER
def highFourierFilter(image, d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(fshift))

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                if dis <= d:
                    transfor_matrix[i,j] = 0
                else:
                    transfor_matrix[i,j] = 1
        return transfor_matrix
    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

img = cv2.imread(path, 0)
new_img = highFourierFilter(img,60)
cv2.imshow('23 HIGH FOURIER TRANSFORM', new_img)
cv2.waitKey()


# 24 ERODE IMAGE
img = cv2.imread(path, 1)
kernel = np.ones((5, 5), 'uint8')
erode_img = cv2.erode(img, kernel, iterations=1)
cv2.imshow('24 ERODED IMAGE', erode_img)
cv2.waitKey()

# 25 DILATE IMAGE
img = cv2.imread(path, 1)
kernel = np.ones((5, 5), 'uint8')
dilate_img = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('25 DILATED IMAGE', dilate_img)
cv2.waitKey(0)



