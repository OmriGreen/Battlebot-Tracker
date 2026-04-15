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

import torch

import clip

import sys

import utils as ut

import time

from PIL import Image

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
        #Initializes OpenAI Clip for image comparison
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.battlebots = None
        self.housebot = None
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
        battlebots["size"] = [] #For identification

        housebot = {}



        bot_id  = 0
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


                roi = pic[ref_y0:ref_y1,ref_x0:ref_x1]
                #Checks if the robot is on the top or bottom of the arena to determine its team
                team = 0
                #Bottom team
                if(centroid[0] >= h/2):
                    team = 1

                #Resizes reference to the same size and then encodes it for CLIP
                ref_img = ut.normalize_ref(roi)
                ref_img = (ref_img * 255).astype(np.uint8) if ref_img.dtype != np.uint8 else ref_img  # ensure uint8
                ref_img = self.preprocess(Image.fromarray(ref_img)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    ref_img = self.model.encode_image(ref_img)


                #Adds all relevant information to the reference images
                battlebots["ref_pic"].append(ref_img)
                battlebots["team"].append(team)
                battlebots["id"].append(bot_id )
                battlebots["loc"].append(centroid)
                battlebots["size"].append(size)

                bot_id  += 1
            #Initializes Housebot Location value
            else:
                if(label == "house_bot"):
                    housebot["loc"] = centroid
                    housebot["size"] = size
        
        #Initializes battlebot and housebot information
        self.battlebots = battlebots
        self.housebot = housebot

    def update_Positions(self,pic,r):
        info = self.extract_data(r)
        identified_battlebot_ids = []
        for centroid, size, label in zip(info["centroid"],info["size"],info["label"]):
            if(label == "battle_bot"):
                #Checks if all robots were identified
                if(len(identified_battlebot_ids) < len(self.battlebots["id"])):
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


                    roi = pic[ref_y0:ref_y1,ref_x0:ref_x1]
                    #Checks if the robot is on the top or bottom of the arena to determine its team
                    team = 0
                    #Bottom team
                    if(centroid[0] >= h/2):
                        team = 1

                    #Resizes robot image to the same size as the reference image then encodes it for CLIP
                    img = ut.normalize_ref(roi)
                    img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img  # ensure uint8
                    img = self.preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        img = self.model.encode_image(img)
                    #Identifies the battlebot and then updates the battlebot's location

                    bot_id  = self.identify_battlebot(img,identified_battlebot_ids)

                    
                    for loc, r_id in enumerate(self.battlebots["id"]):
                        if bot_id  == r_id:
                            self.battlebots["loc"][loc] = centroid
                            break
                
                    identified_battlebot_ids.append(bot_id )

                #Updates Housebot Location value since there is always one battlebot
            else:
                if(label == "house_bot"):
                    self.housebot["loc"] = centroid
                    self.housebot["size"] = size

    def identify_battlebot(self,pic,identified_battlebot_ids):
        robot_id  = -1
        robot_similarity = -1
        # Goes through battlebot data and finds the most similar one
        for id, ref_pic in zip(self.battlebots["id"], self.battlebots["ref_pic"]):
            #Doesn't examine already identified robots
            if(id not in identified_battlebot_ids):
                #Finds the similarity between the reference picture and the battlebot
                similarity = (pic @ ref_pic.T).item()

                #Checks if it is the greatest similarity
                if robot_similarity < similarity:
                    robot_id = id
                    robot_similarity = similarity
                
        return robot_id

    #Returns the data on the battlebots and housebot for identification and graphing
    def retrieve_data(self):
        return self.battlebots, self.housebot
    

                    







        

    

        
        



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
            start = time.perf_counter() # More precise than time.time()
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

            #Checks robot location every 125 ms
            if(calc_Time(fps,frames_AI) >= 125 or frames == 0):
                results = model(top_view,conf = 0.3,verbose=False)[0]
                frames_AI = 0

            #Shows results on video
            annotated_frame = results.plot()

            #Displays the frame
            # cv.imshow("Robot Detection", annotated_frame)

            # Robot Identifier =========================================================================================
            if(frames == 0):
                identifier = robot_Identifier(top_view,results)


            
            #Extracts centroid from the top view FOR TESTING!!!!!!!!!!
            info = identifier.extract_data(results)

            #Identifies robot position
            identifier.update_Positions(top_view,results)
            
            #Shows identified robot IDs on top_view
            identified = top_view

            #gets robot data
            battlebots, housebot = identifier.retrieve_data()
            #Draws identification data for the housebot
            cv.circle(identified,housebot["loc"],int(housebot["size"]/2), (0,255,0), 5)
            
            #Draws identification data for the battlebots
            for l,d,t,rob_id in zip(battlebots["loc"],battlebots["size"],battlebots["team"],battlebots["id"]):
                if(t == 0):
                    cv.circle(identified,l, int(d/2), (0,0,255), 5)
                else:
                    cv.circle(identified,l, int(d/2), (255,0,0), 5)
                

            end = time.perf_counter()
            print(f"Inference Time: {end - start:.4f} seconds")
            # for l,c,d in zip(info["label"],info["centroid"],info["size"]):
            #     if(l == "battle_bot"):
            #         cv.circle(identified,c, int(d/2), (0,0,255), 5)
            #     else:
            #         cv.circle(identified,c, int(d/2), (0,255,0), 5)
            cv.imshow("Centroid Detection", identified)


            

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            frames += 1
            frames_AI +=1

        


 

    recording = 0