import numpy as np
import cv2

class Playground(object):
    
    def cv_contour_extraction(self):
        image = cv2.imread("data/pic.jpg")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        ret, threshold = cv2.threshold(gray_image,127,255,0)
        ret2, threshold2 = cv2.threshold(gray_image,10,255,cv2.THRESH_BINARY)
        adapt_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        im2, contours, hierarchy = cv2.findContours(adapt_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(contours)
        print(hierarchy)
        
        
        cv2.drawContours(im2, contours, -1, (0,255,0),3)
        
        cnt = contours[4]
        #cv2.imshow("Image", image)
        #cv2.imshow("Gray Image", gray_image)
        #cv2.imshow("Threshold 2", threshold)
        #cv2.imshow("Threshold Adaptive", adapt_threshold)
        cv2.imshow("Contours", im2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def preprocess(self, image):
        gray = cv2.cvtColor(image,cv2.COLOR_BG2GRAY)
        blur = cv2.GaussianBlur(gray,(1,1),1000)
        flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
        return thresh
    
    def find_contours(self, image):
        contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours
        
if __name__ == '__main__':
    playground = Playground()
    playground.cv_contour_extraction()
    
    image = cv2.imread("data/pic.jpg")
    processed_image = playground.preprocess(image)
    contours = playground.find_contours(processed_image)
    
    image_copy = processed_image.copy()
    
    cv2.drawContours(image_copy,contours,0,(0,255,0), 5)
    cv2.imshow("Extracted Contours",image_copy)