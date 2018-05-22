import numpy as np
import cv2

image = cv2.imread("data/pic.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, threshold = cv2.threshold(gray_image,127,255,0)
ret2, threshold2 = cv2.threshold(gray_image,10,255,cv2.THRESH_BINARY)
adapt_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
im2, contours, hierarchy = cv2.findContours(threshold2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(contours)
print(hierarchy)

cv2.drawContours(im2, contours, -1, (0,255,0),3)

cnt = contours[4]
cv2.drawContours(gray_image,[cnt],0,(0,255,0),3)

cv2.imshow("Image", image)
cv2.imshow("Gray Image", gray_image)
cv2.imshow("Threshold 2", threshold)
cv2.imshow("Threshold Adaptive", adapt_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
