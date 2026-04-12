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
"""
Allows the software to detect which robot is which, i.e. what "Team" each robot is on
"""
class robot_Tracker:
    """
    Initializes the Robot Tracker and stores information on tracking the robots
        Inputs:
            r: results from the CV model
            pic: an initial picutre of the arena
    """
    def __init__(self,r,pic):
        self.battlebots = None
        self.housebot = None



    """
    Extracts general data from the Computer Vision Model

    Input:
        r : resulting data from the CV model
        pic: The image that the CV Model is analyzing
    Output:
        battlebot_data: A list of data containing the following information
            ref_pic: A xyxy coordinate that will be used as an area that will be normalized for matching to an original image
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
        info["radius"] = []
        info["ref_pic"] = []

        for bBox, class_id in zip(xyxy, cls):
            # Generic data for both battlebots and housebots (allows for location tracking)
            label = names[int(class_id)]
            centroid = self.calc_Centroid(bBox)
            radius = self.calc_Radius(bBox)
            info["label"].append(label)
            info["centroid"].append(centroid)
            info["radius"].append(radius)

            #Ref Pic Information (Only Really relevant for battlebots)
            info["ref_pic"].append([centroid[0]-radius, centroid[1]-radius,
                                centroid[0]+radius, centroid[1]+radius])
        

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
    Calculates the maximum radius of the robot for later matching
        Input:
            bBox: The bounding box
        Output:
            Radius: The radius of the area around the robot 
    """
    def calc_Radius(self,bBox):
        return max(abs(int((bBox[0] - bBox[2]) / 2)),
        abs(int((bBox[1] - bBox[3]) / 2)))

   



        

    

        
        



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

            #Checks robot location every 50 ms
            if(calc_Time(fps,frames_AI) >= 50 or frames == 0):
                results = model(top_view,conf = 0.5,verbose=False)[0]
                frames_AI = 0

            

            # Robot Tracker =========================================================================================
            if(frames == 0):
                tracker = robot_Tracker(results,top_view)

            #Extracts centroid from the top view FOR TESTING!!!!!!!!!!
            info = tracker.extract_data(results)
            
            #Shows boxed out shell on top_view
            box_View = top_view

            for l,c,r,p in zip(info["label"],info["centroid"],info["radius"],info("ref_pic")):
                if(l == "battle_bot"):
                    cv.rectangle(box_View,())
                    # cv.circle(centroid_View,c, r, (0,0,255), 5)
                else:
                    cv.circle(centroid_View,c, r, (0,255,0), 5)
            cv.imshow("Centroid Detection", centroid_View)


            

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            frames += 1
            frames_AI +=1

        print(f"Finished processing video {vid_counter}/{len(video_names)}: {video}")


 

    recording = 0