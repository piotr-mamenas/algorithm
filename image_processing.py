import numpy as np
import cv2

class Playground(object):
    
    def cv_contour_extraction(self):
        image = cv2.imread("data/pic.jpg")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        ret, threshold = cv2.threshold(gray_image,127,255,0)
        ret2, threshold2 = cv2.threshold(gray_image,10,255,cv2.THRESH_BINARY_INV)
        adapt_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        im2, contours, hierarchy = cv2.findContours(adapt_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        print(contours)
        print(hierarchy)
        
        cv2.drawContours(im2, contours, -1, (0,255,0),3)
        
        #cv2.imshow("Image", image)
        #cv2.imshow("Gray Image", gray_image)
        #cv2.imshow("Threshold 2", threshold)
        #cv2.imshow("Threshold Adaptive", adapt_threshold)
        self.show_and_wait(im2, "Contours")
    
    def show_and_wait(self, image, image_text):
        cv2.imshow(image_text, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(gray,(1,1),1000)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        dilated = cv2.dilate(thresh, kernel, iterations = 13)
        
        return dilated
    
    def find_contours(self, image):
        image, contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)
        #contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return contour
        
if __name__ == '__main__':
    playground = Playground()
    playground.cv_contour_extraction()
    
    #image = cv2.imread("data/pic.jpg")
    image = cv2.imread("data/pic3.jpeg")
    processed_image = playground.preprocess(image)
    contours = playground.find_contours(processed_image)
    
    image_copy = processed_image.copy()
    
    concrete_contour = contours[1]
    
    cv2.drawContours(image_copy,[concrete_contour],0,(0,255,0), 5)
    playground.show_and_wait(image_copy,"Extracted Contours")