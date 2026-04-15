import numpy as np
import cv2 as cv
import gradio as gr
import os

import numpy as np
import cv2 as cv
import gradio as gr
import os
import math

from ultralytics import YOLO

import sys



#Resizes the image to 540x960
def normalize_img(newImg):
    return cv.resize(newImg,(960,540))

#Resizes reference image to the same square size
def normalize_ref(ref):
    resized = cv.resize(ref, (50, 50))
    return resized.astype(np.uint8)
        

#Finds the intersection point of 2 lines
#Input: l1 = ((x0,y0),(x1,y1)), l2 = ((x0,y0),(x1,y1))
def find_intersection(l1,l2):
    xdiff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
    ydiff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*l1), det(*l2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return round(x), round(y)



class transform_img:
    def __init__(self, img, buffer = 50):
        self.img = img
        self.buffer = 50

   
   #Detects the vertices of the square for transforming the image to a top down view
    def detect_Vertices(self):

        #Checks if the image is valid for training and autonomy
        validImg = True

        #Stores all vertices
        vertices = []

        #Embeds the image into a black square for better transformations
        h,w = self.img.shape[:2]

        black_sq = np.zeros((h+250,w+250,3),np.uint8)

        x_offset = (250)//2
        y_offset = (250) //2

        black_sq[y_offset:y_offset+h,x_offset:x_offset+w] = self.img

        # convert the image into grayscale
        grayScale = cv.cvtColor(black_sq,cv.COLOR_BGR2GRAY)

        #Apply Guassian Blue
        blur = cv.GaussianBlur(grayScale,(9,9),1.4)

        #Detect Canny Edges
        edges= cv.Canny(blur,threshold1=100,threshold2=200)

        #Uses Hough transform to get the largest lines
        lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

        # Gets the cartesian coordinates of the lines
        radial_Lines = cv.cvtColor(grayScale, cv.COLOR_GRAY2BGR)
        cartesian_Lines = []
        top_lines = []
        right_lines = []
        top_line_data = []
        left_line_data = []
        right_line_data = []
        if lines is not None:

            #Gets vertical and horizontal lines for intersection tests
            v_left = ((125,125),(125,grayScale.shape[0]-125))
            # cv.line(radial_Lines,v_left[0],v_left[1],(0,0,255),3,cv.LINE_AA)

            v_right = ((grayScale.shape[1]-125,125),(grayScale.shape[1]-125,grayScale.shape[0]-125))
            # cv.line(radial_Lines,v_right[0],v_right[1],(0,0,255),3,cv.LINE_AA)

            h_bottom = ((0,grayScale.shape[0]-125), (grayScale.shape[1],grayScale.shape[0]-125))
            cv.line(radial_Lines,h_bottom[0],h_bottom[1],(0,255,255),3,cv.LINE_AA)

            h_top = ((125,125), (grayScale.shape[1]-125,125))
            # cv.line(radial_Lines,h_top[0],h_top[1],(0,0,255),3,cv.LINE_AA)



            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho

                
                
                #Gets the initial points of the coordinate
                pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))

                line = (pt1,pt2)

                # Finds intersection point with the edges of the image
                try:
                    inter_top = find_intersection(h_top,line)
                    inter_bottom = find_intersection(h_bottom,line)
                    inter_left = find_intersection(v_left,line)
                    inter_right = find_intersection(v_right,line)                    
                    
                    # Detects left lines for the transformation
                    if(inter_top[0] > 125 and inter_top[0] <= black_sq.shape[0]-125 and inter_bottom is not None):
                        #Checks if the line is  correct and it isn't reading a false line
                        if(inter_top[0] > inter_bottom[0]):
                            left_line_data.append((inter_top,inter_bottom))
                    else:
                        #Detects right lines 
                        if(inter_top[0] < inter_bottom[0] and max(inter_top) <= black_sq.shape[1] and min(inter_top) > 0 and  inter_bottom[0] > 0): 
                            right_line_data.append((inter_top,inter_bottom))
                        #Detects the top line
                        else:
                            #Eliminates and "Bounding lines"
                            # Eliminates vertical lines and lines below a half of the height of the arena
                            if((line[0][0]) != line[1][0] and inter_left[1] < black_sq.shape[0]//2):
                                # A double check to determime that the top image doesn't exactly match the top line
                                if(h_top[0][1] != inter_left[1] and h_top[0][1] != inter_right[1]):
                                    top_line_data.append((inter_left,inter_right))
                except:
                    continue


                            
            #Processes all left lines to find the ideal left line by finding the rightmost top and bottom points
            left_top = (-5,0)
            left_bottom = (-5,0)
            for line_data in left_line_data:
                if line_data[0][0] > left_top[0]:
                    left_top = line_data[0]
                if line_data[1][0] > left_bottom[0]:
                    left_bottom = line_data[1]
            left_line = (left_top,left_bottom)
            cv.line(radial_Lines,left_line[0],left_line[1],(0,0,255),3,cv.LINE_AA)

            #Processes all right lines to find the ideal right line by finding the leftmost top and bottom points
            right_top = (max(black_sq.shape),0)
            right_bottom = (max(black_sq.shape),0)
            for line_data in right_line_data:
                if line_data[0][0] < right_top[0]:
                    right_top = line_data[0]
                if line_data[1][0] < right_bottom[0]:
                    right_bottom = line_data[1]
            right_line = (right_top,right_bottom)
            cv.line(radial_Lines,right_line[0],right_line[1],(0,255,0),3,cv.LINE_AA)

            # Processes all top lines to find the ideal one by finding the lowest one that fits the criteria
            top_left = (0,0)
            top_right = (0,0)
            print(top_line_data)
            for line_data in top_line_data:
                if(line_data[0][1] > top_left[1]):
                    top_left = line_data[0]
                if(line_data[1][1] > top_right[1]):
                    top_right = line_data[1]
            top_line = (top_left,top_right)
            cv.line(radial_Lines,top_line[0],top_line[1],(255,0,0),3,cv.LINE_AA)

            # Finds the intersection points between all points
            top_left = find_intersection(top_line, left_line)
            top_right = find_intersection(top_line,right_line)
            bottom_left = find_intersection(h_bottom, left_line)
            bottom_right = find_intersection(h_bottom,right_line)

            #Draws dots for visualization
            cv.circle(radial_Lines,top_left, 10, (255,255,255), -1)
            cv.circle(radial_Lines,top_right, 10, (255,255,255), -1)
            cv.circle(radial_Lines,bottom_left, 10, (255,255,255), -1)
            cv.circle(radial_Lines,bottom_right, 10, (255,255,255), -1)
            print("++++++++++++++")
            print(top_left)
            print(top_right)
            print(bottom_left)
            print(bottom_right)
            print("++++++++++++++")

            # Last checks for validity
            try:
                if(min(top_left) < 0 or min(top_right) < 0 or min(bottom_left) < 0 or min(bottom_right) < 0):
                    print("INVALID IMAGE")
                    vertices = None
                else:
                    vertices = [top_left, top_right, bottom_right, bottom_left]                    
            except:
                print("INVALID IMAGE")
                vertices = None

        #Find intersection points between hough lines
        return vertices, radial_Lines

    def setupTransformMatrix(self, squarePts):
        if squarePts is None or len(squarePts) < 4:
            print("setupTransformMatrix: Not enough points to define a square.")
            return None
        
        self.squarePts = np.array(squarePts, dtype="float32")

        b = self.buffer
        dstPts = np.array([
            [b,       b        ],  # top-left
            [960 + b, b        ],  # top-right
            [960 + b, 960 + b  ],  # bottom-right
            [b,       960 + b  ]   # bottom-left
        ], dtype="float32")

        self.transformMatrix = cv.getPerspectiveTransform(self.squarePts, dstPts)
        return self.transformMatrix
        
    def transform_img(self, img, vertices):
        M = self.setupTransformMatrix(vertices)
        if M is not None:
            h, w = img.shape[:2]
            black_sq = np.zeros((h + 250, w + 250, 3), np.uint8)
            x_offset = 250 // 2
            y_offset = 250 // 2
            black_sq[y_offset:y_offset + h, x_offset:x_offset + w] = img

            output_size_x = 960 + 2 * self.buffer  # left and right buffer
            output_size_y = 960 + self.buffer       # top buffer only, no bottom gap
            warped = cv.warpPerspective(black_sq, M, (output_size_x, output_size_y))
        else:
            warped = None
        return warped
