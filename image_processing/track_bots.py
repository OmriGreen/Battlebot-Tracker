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

import utils as ut

sys.path.append('../image_processing')

#Calculates the time (in milliseconds) based on the frame count and fps of the video
def calc_Time(fps,frame_count):
    return (frame_count/fps)*1000


# Robot Data
class robot_Identifier:
    """
    Initializes the Robot Identifier and stores information on identifying the robots
        Inputs:
            pic: an initial picture of the arena
        Outputs:
            self.battlebots: A dictionary containing the id, reference image, team, and current location of each battlebot
                id: a list of numeric IDs for each battlebot
                ref_pic: A list of reference pictures of each battlebot
                team: A numeric value categorizing what team the battlebot is on
                loc: The robot's current location

            self.housebot_loc: Current location of the housebot
    """
    def __init__(self,pic,r):
        self.battlebots = None
        self.housebot_loc = None
        self.init_identifier(pic, r)



    """
    Extracts general data from the Computer Vision Model

    Input:
        r : resulting data from the CV model
        pic: The image that the CV Model is analyzing
    Output:
        battlebot_data: A list of data containing the following information
            ref_pic: A picture of the battlebot to match to previous reference images
            loc: (x,y) tuple of the location in x,y coordinates in meters
        housebot_loc: (x,y) tuple of the robot's location in x,y coordinates in meters
    """
    def extract_data(self,r):
        #Extracts data from init_data
        boxes = r.boxes

        # Box Coordinates
        xyxy = boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]

        #Labels
        cls = boxes.cls.cpu().numpy()
        names = r.names  # dictionary mapping id → label
        labels = [names[int(c)] for c in cls]


        #Raw information
        info = {}
        info["centroid"] = []
        info["label"] = []
        info["size"] = []
        for bBox, class_id in zip(xyxy, cls):
            label = names[int(class_id)]
            info["label"].append(label)
            info["centroid"].append(self.calc_Centroid(bBox))
            info["size"].append(self.calc_Size(bBox))
        return info

    """
    Calculates the centroid of a robot

        Input: 
            bBox: The bounding box 
        Output:
            (x,y): The centroid of the bounding box
    """ 
    def calc_Centroid(self,bBox):
        return (
    int((bBox[0] + bBox[2]) / 2),
    int((bBox[1] + bBox[3]) / 2)
    )
    """
    Finds the maximum size of the robot based on the bounding box
        Input:
            bBox: The bounding box
        Output:
            size: The maximum size of the robot in pixels
    """
    def calc_Size(self,bBox):
        width = bBox[2] - bBox[0]
        height = bBox[3] - bBox[1]
        return int(max(width, height))

    """
        Initializes the identifier with all relevant information for tracking the robots
            Input:
                img: The image that the CV model is analyzing
            Output:
                battlebots: A dictionary containing reference images and labels of the battlebots as well as the team of the robot, location, velocity information for each battlebot 
                    ref_pic: A picture of the battlebot to match to previous reference images
                    team: The team of the battlebot (0 or 1)(top or bottom)
                    id: The numeric id of the battlebot (battle_bot)
                    loc: The current location of the robot
    """
    def init_identifier(self,pic, r):
        info = self.extract_data(r)
        battlebots = {}
        battlebots["ref_pic"] = []
        battlebots["team"] = []
        battlebots["id"] = []
        battlebots["loc"] = []




        id = 0
        for centroid, size, label in zip(info["centroid"],info["size"],info["label"]):
            if(label == "battle_bot"):
                #Gets height and width of image for analysis
                h, w = pic.shape[:2]


                #Extracts an image of the robot for later reference
                ref_y0 = int(centroid[1]-size/2)
                ref_y1 = int(centroid[1] + size/2)
                ref_x0 = int(centroid[0]-size/2)
                ref_x1 = int(centroid[0] + size/2)

                #Screens robot information to prevent the data from going out of bounds

                #Checks if x0 is out of bounds
                if ref_x0 < 0:
                    ref_x1 -= ref_x0
                    ref_x0 = 0

                #Checks if y0 is out of bounds
                if ref_y0 < 0:
                    ref_y1 -= ref_y0
                    ref_y0 = 0

                #Checks is x1 is out of bounds
                if ref_x1 > w:
                    ref_x0 -= (ref_x1-w)
                    ref_x1 = w

                #Checks if y1 is out of bounds
                if ref_y1 > h:
                    ref_y0 -= (ref_y1-h)
                    ref_y1 = h


                print(ref_x0)
                print(ref_x1)
                print(ref_y0)
                print(ref_y1)


                roi = pic[ref_y0:ref_y1,ref_x0:ref_x1]
                print("ROI RETRIEVED")

                #Checks if the robot is on the top or bottom of the arena to determine its team
                team = 0
                #Bottom team
                if(centroid[0] >= h/2):
                    team = 1

                #Resizes reference to the same size
                ref_img = ut.normalize_ref(roi)

                #Adds all relevant information to the reference images
                battlebots["ref_pic"].append(ref_img)
                battlebots["team"].append(team)
                battlebots["id"].append(id)
                battlebots["loc"].append(centroid)

                id += 1
            #Initializes Housebot Location value
            else:
                self.housebot_loc = centroid
        
        #Initializes battlebot information
        self.battlebots = battlebots
                

                    







        

    

        
        



if __name__ == '__main__':

    model = YOLO("detector.pt")
    cv.namedWindow('image')
    fourcc = cv.VideoWriter_fourcc(*'MJPG')  


    # gets the names of all videos in the Test_Videos folder
    video_names = os.listdir('Test_Videos')
    video_names = [name for name in video_names if name.endswith('.mp4')]

    vid_counter = 0
    pic_counter = 0

    for video in video_names:
        results = None
        pic_counter = 0

        frames = 0
        frames_AI = 0
        cap = cv.VideoCapture(f'Test_Videos/{video}')
        if not cap.isOpened():
            print(f"Cannot open video {video}")
            exit()
        
        fps = cap.get(cv.CAP_PROP_FPS)



        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Can't receive frame (stream end?). Exiting ...")
                break

            frame = ut.normalize_img(frame)
            #Shows the raw video ==================================================================
            # cv.imshow('Raw Video', frame)



            #Shows the transformed video (No AI Detection) ============================================================

            #Gets the matrix transformation from the initial frame

            if(frames == 0):
                tI = ut.transform_img(frame)
                detectVertices = tI.detect_Vertices()
                vertices = detectVertices[0]


             #Transforms the frame to a top down view
            top_view = tI.transform_img(frame,vertices)

            # cv.imshow("Top Down View", top_view)

            #Robot Detection ==========================================================================================

            #Checks robot location every 20 ms
            if(calc_Time(fps,frames_AI) >= 20 or frames == 0):
                results = model(top_view,conf = 0.3,verbose=False)[0]
                frames_AI = 0

            #Shows results on video
            annotated_frame = results.plot()

            #Displays the frame
            cv.imshow("Robot Detection", annotated_frame)

            # Robot Identifier =========================================================================================
            if(frames == 0):
                identifier = robot_Identifier(top_view,results)

                print(identifier.battlebots["id"])
                print(identifier.battlebots["team"])

                # for pic,id in zip(identifier.battlebots["ref_pic"], identifier.battlebots["id"]):
                #     cv.imshow(f'Battlebot {str(id)}',pic)
                #     cv.waitKey(0)


            # #Extracts centroid from the top view FOR TESTING!!!!!!!!!!
            # info = identifier.extract_data(results)
            
            # #Shows centroid on top_view
            # centroid_View = top_view

            # for l,c,d in zip(info["label"],info["centroid"],info["size"]):
            #     if(l == "battle_bot"):
            #         cv.circle(centroid_View,c, int(d/2), (0,0,255), 5)
            #     else:
            #         cv.circle(centroid_View,c, int(d/2), (0,255,0), 5)
            # cv.imshow("Centroid Detection", centroid_View)
            

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            frames += 1
            frames_AI +=1

        


 

    recording = 0