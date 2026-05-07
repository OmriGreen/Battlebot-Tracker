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

import sys

import utils as ut

import time

from PIL import Image

import math 

sys.path.append('../image_processing')

#Calculates the time (in milliseconds) based on the frame count and fps of the video
def calc_Time(fps,frame_count):
    return (frame_count/fps)*1000


"""
extract_loc:
    Description: Extracts the location of all the robots from the result of the YOLO network for later processing
    Input:
        r : resulting data from the CV model
    Output:
        battlebot_loc: A list of data containing the following information
            loc: (x,y) tuple of the location in x,y coordinates in pixels
            id: A numerical ID representing the robot
        housebot_loc:  (x,y) tuple of the location in x,y coordinates in pixels
"""
def extract_loc(r):
    #Extracts data from init_data
    boxes = r.boxes

    # Box Coordinates
    xyxy = boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]

    #Robot Ids
    idents = boxes.id.int().cpu().tolist()

    #Labels
    cls = boxes.cls.cpu().numpy()
    names = r.names  # dictionary mapping id → label
    labels = [names[int(c)] for c in cls]

    #Extracting Battlebot locations
    battlebot_data = {}
    battlebot_data["loc"] = []
    battlebot_data["id"] = []

    #Extracting Housebot location
    housebot_loc = []

    #Extracting Data
    for bound, label,ident in zip(xyxy,labels,idents):
        #Calculates the location
        x_loc = round((bound[0]+bound[2])/2)
        y_loc = round((bound[1]+bound[3])/2)

        loc = (x_loc,y_loc)

        #If the robot detected is a battlebot
        if(label == "battle_bot"):
            battlebot_data["loc"].append(loc)
            battlebot_data["id"].append(ident)
           
        #If the robot detected is a housebot
        if(label == "house_bot"):
            housebot_loc = loc
         

    return battlebot_data, housebot_loc
        
"""
find_scale: Calculates the scale from pixels to feet from the top down view
    Inputs: 
        image_size: (width,height)
            width: the initial width of the image
            height: the initial height of the image
        buffer: the initial offset size in pixels from Utils (125 px)
        nn_size: (width,height)
            width: the initial width of the image
            height: the initial height of the image
        arena_size: a double containing the width/size of the arena in feet (NHRL arena is a square measured in feet can be changed)
    Outputs:
        scale: a double when multiplied by the pixel value goes to the actual location
        arena_bounds: (x1,y1,x2,y2) for determining if a location is outside the arena
"""
def find_scale(initial_size = (1060,1010),buffer=50,nn_size = (400,381),arena_size = 8):
    #Gets the initial arena_bounds not from the nn_scale
    init_x1 = buffer
    init_x2 = initial_size[0]-buffer
    init_y1 = 0
    init_y2 = initial_size[1]-buffer

    init_bounds = [init_x1,init_y1,init_x2,init_y2]

    arena_width = abs(init_x1-init_x2)
    arena_height = abs(init_y1-init_y2)


    #img_scale: goes from the first image to the second image
    img_scale = (nn_size[0]/initial_size[0])

    #Finds the arena bounds in pixels
    arena_bounds = []
    for val in init_bounds:
        arena_bounds.append(round(val*img_scale))
    
    #Height of boxed in arena divided by the height of the arena
    scale = arena_size/arena_bounds[3]

    return scale, arena_bounds

"""
find_real_loc: Calculates the real location of an object based on its pixel values
    Input:
        loc: an x,y tuple containing the location of the detected robot in pixels
        arena_bounds: a xyxy tuple containing the information about the bounds of the arean
        scale: a pixel to ft convertor
    Output:
        real_loc: an x,y tuple containing the location of the detected robot in feet
"""
def find_real_loc(loc,arena_bounds,scale):
    real_loc = (loc[0]-arena_bounds[0],loc[1]-arena_bounds[1])
    real_loc = (real_loc[0]*scale, real_loc[1]*scale)
    real_loc = (round(real_loc[0],2),round(real_loc[1],2))
    return real_loc

"""
process_loc: Processes the location of the robot to get the real location for all housebots and battlebots
    Input:
        battlebot_loc_px: A list of data containing the following information
            loc: (x,y) tuple of the location in x,y coordinates in pixels
            id: A numerical ID representing the robot
        housebot_loc_px:  (x,y) tuple of the location in x,y coordinates in pixels
        scale: a numerical value containing the ratio between pixels and ft
        arena_bounds: The bounds of the arena in feet
    Output:
        battlebot_loc: A list of data containing the following information
            loc_px: (x,y) tuple of the location in x,y coordinates in pixels
            loc_ft: (x,y) tuple of the location in x,y coordinates in feet
            id: A numerical ID representing the robot
        housebot_loc: A dictionary containing the following information
            loc_px: (x,y) tuple of the location in x,y coordinates in pixels
            loc_ft: (x,y) tuple of the location in x,y coordinates in feet
"""
def process_loc(battlebot_loc_px,housebot_loc_px,scale,arena_bounds):
    
    #Battlebot Processing
    battlebot_loc = {}
    battlebot_loc["loc_px"] = battlebot_loc_px["loc"]
    battlebot_loc["id"] = battlebot_loc_px["id"]
    battlebot_loc["loc_ft"] = []

    for loc_px in battlebot_loc_px["loc"]:
        battlebot_loc["loc_ft"].append(find_real_loc(loc_px, arena_bounds,scale))

    #housebot Processing
    try:
        housebot_loc = {}
        housebot_loc["loc_px"] = housebot_loc_px
        housebot_loc["loc_ft"] = find_real_loc(housebot_loc_px,arena_bounds,scale)
    except:
        housebot_loc = None
   

    return battlebot_loc, housebot_loc

"""
velocity_detector:A class for calculating robot velocity and direction of travel
    Storage:
        dt: The time difference between data in milliseconds
        battlebot_history: a dictionary containing data on detected battlebots, if their id is not detected they are then deleted from the history
            {z}: A dictionary representing the robots detected with the id z that contains the following:
                prev_loc_ft: The previous location of the robot in feet
                prev_loc_px: The previous location of the robot in pixels
                curr_loc_ft: The current location of the robot in ft 
                curr_loc_px: The current location of the robot in pixels
        housebot_history: a dictionary containing data on the housebot which contains the following:
            prev_loc_ft: The previous location of the robot in feet
            prev_loc_px: The previous location of the robot in pixels
            curr_loc_ft: The current location of the robot in ft 
            curr_loc_px: The current location of the robot in pixels
    Functions:
        update_hist: updates the current location of every detected robot into the dataset
        calc_Kinematics: calculates the velocity and heading of the robots in the arena
"""
class velocity_detector:
    def __init__(self,dt):
        self.dt = dt/1000
        self.battlebot_history = {}
        self.housebot_history = {}
        self.housebot_Kinematics = {}
    
    """
    update_hist: updates the current location of every detected robot into the dataset
        Inputs:
            battlebot_loc: A list of data containing the following information
                loc_px: (x,y) tuple of the location in x,y coordinates in pixels
                loc_ft: (x,y) tuple of the location in x,y coordinates in feet
                id: A numerical ID representing the robot
            housebot_loc: A dictionary containing the following information
                loc_px: (x,y) tuple of the location in x,y coordinates in pixels
                loc_ft: (x,y) tuple of the location in x,y coordinates in feet
        Outputs:
            None
    """
    def update_hist(self,battlebot_loc,housebot_loc):
        #Clears battlebot_history of any information that is not relevant

        bad_keys = []
        for key in self.battlebot_history.keys():
            if key not in battlebot_loc["id"]:
                bad_keys.append(key)

        for key in bad_keys:
            del self.battlebot_history[key]

        
        #Updates battlebot history
        for loc_px,loc_ft,key in zip(battlebot_loc["loc_px"],battlebot_loc["loc_ft"],battlebot_loc["id"]):
            #Creates a key if it doesn't exist with the assumption that the robot is at a standstill
            if key not in self.battlebot_history.keys():
                self.battlebot_history[key] = {}
                self.battlebot_history[key]["prev_loc_ft"] = loc_ft
                self.battlebot_history[key]["prev_loc_px"] = loc_px
                self.battlebot_history[key]["curr_loc_ft"] = loc_ft
                self.battlebot_history[key]["curr_loc_px"] = loc_px    
                self.battlebot_history[key]["theta"] = -10
            #Otherwise it updates the previous and current locations
            else:
                #Updates previous information
                self.battlebot_history[key]["prev_loc_ft"] = self.battlebot_history[key]["curr_loc_ft"]
                self.battlebot_history[key]["prev_loc_px"] = self.battlebot_history[key]["curr_loc_px"]
                #Updates new information
                self.battlebot_history[key]["curr_loc_ft"] = loc_ft
                self.battlebot_history[key]["curr_loc_px"] = loc_px    
        
        #Updates housebot history

        #Populates data if there is not history with the assumption it starts at a standstill
        if "prev_loc_ft" not in self.housebot_history.keys() or self.housebot_history["prev_loc_ft"][0] == -10:
            try:
                self.housebot_history["prev_loc_ft"] = housebot_loc["loc_ft"]
                self.housebot_history["prev_loc_px"] = housebot_loc["loc_px"]
                self.housebot_history["curr_loc_ft"] = housebot_loc["loc_ft"]
                self.housebot_history["curr_loc_px"] = housebot_loc["loc_px"]
                self.housebot_history["theta"] = -10
            except:
                self.housebot_history["prev_loc_ft"] = (-10,-10)
                self.housebot_history["prev_loc_px"] = (-10,-10)
                self.housebot_history["curr_loc_ft"] = (-10,-10)
                self.housebot_history["curr_loc_px"] = (-10,-10)
                self.housebot_history["theta"] = -10
        #Otherwise it updates the information
        else:
            try:
                #Updates previous info
                self.housebot_history["prev_loc_ft"] = self.housebot_history["curr_loc_ft"]
                self.housebot_history["perv_loc_px"] = self.housebot_history["curr_loc_px"]
                #Updates current information
                self.housebot_history["curr_loc_ft"] = housebot_loc["loc_ft"]
                self.housebot_history["curr_loc_px"] = housebot_loc["loc_px"]
            except:
                pass

    """
    calc_Kinematics: calculates the velocity and heading of the robots in the arena
        Inputs:
            None
        Outputs:
            battlebot_Kinematics: A list of data containing the following information
                loc_px: (x,y) tuple of the location in x,y coordinates in pixels
                loc_ft: (x,y) tuple of the location in x,y coordinates in feet
                theta: The robot's current direction of travel
                vel_px: The robot's velocity in pixels/s (for display)
                vel_ft: The robot's velocity in ft/s
            housebot_Kinematics: A dictionary containing the following data about the housebot
                loc_px: (x,y) tuple of the location in x,y coordinates in pixels
                loc_ft: (x,y) tuple of the location in x,y coordinates in feet
                theta: The robot's current direction of travel
                vel_px: The robot's velocity in pixels/s (for display)
                vel_ft: The robot's velocity in ft/s
    """
    def calc_Kinematics(self):
        #Calculate housebot velocity and orientation ==================================================
        #Real Velocity Calculations -----------------------------------------------------
        #Current and previous postion
        prev_x_ft = self.housebot_history["prev_loc_ft"][0]
        prev_y_ft = self.housebot_history["prev_loc_ft"][1]

        curr_x_ft = self.housebot_history["curr_loc_ft"][0]
        curr_y_ft = self.housebot_history["curr_loc_ft"][1]

        #Velocity Calculations
        housebot_dx_ft = curr_x_ft-prev_x_ft
        housebot_dy_ft = curr_y_ft-prev_y_ft
        housebot_vel_ft = round(math.sqrt(housebot_dx_ft**2 + housebot_dy_ft**2)/self.dt,2)

        #Pixel Velocity (for display/other conversions) ---------------------------------------------
        prev_x_px = self.housebot_history["prev_loc_px"][0]
        prev_y_px = self.housebot_history["prev_loc_px"][1]

        curr_x_px = self.housebot_history["curr_loc_px"][0]
        curr_y_px = self.housebot_history["curr_loc_px"][1]

        #Current and Previous positions
        
        housebot_dx_px = curr_x_px-prev_x_px
        housebot_dy_px = curr_y_px-prev_y_px
        housebot_vel_px = round(math.sqrt(housebot_dx_px**2 + housebot_dy_px**2)/self.dt)

        #Calculates theta if the robot is moving(pixel values are probably slightly more accurate)
        if housebot_vel_px > 0:
            housebot_theta = math.atan2(housebot_dy_px,housebot_dx_px)
            self.housebot_history["theta"] = housebot_theta
        else:
            housebot_theta = self.housebot_history["theta"]

        #Storing housebot info
        housebot_Kinematics = {}
        housebot_Kinematics["loc_ft"] = self.housebot_history["curr_loc_ft"]
        housebot_Kinematics["loc_px"] = self.housebot_history["curr_loc_px"]
        housebot_Kinematics["theta"] = housebot_theta
        housebot_Kinematics["vel_px"] = housebot_vel_px
        housebot_Kinematics["vel_ft"] = housebot_vel_ft

     

        # Calculate velocity and orientation for the detected battlebots

        battlebot_Kinematics = {}
        battlebot_Kinematics["loc_ft"] = []
        battlebot_Kinematics["loc_px"] = []
        battlebot_Kinematics["theta"] = []
        battlebot_Kinematics["vel_px"] = []
        battlebot_Kinematics["vel_ft"] = []

        for key in self.battlebot_history.keys():
            battlebot_data = self.battlebot_history[key]
            
            #Real Velocity Calculations -----------------------------------------------------
            #Current and previous postion
            prev_x_ft = battlebot_data["prev_loc_ft"][0]
            prev_y_ft = battlebot_data["prev_loc_ft"][1]

            curr_x_ft = battlebot_data["curr_loc_ft"][0]
            curr_y_ft = battlebot_data["curr_loc_ft"][1]

            #Velocity Calculations
            battlebot_dx_ft = curr_x_ft-prev_x_ft
            battlebot_dy_ft = curr_y_ft-prev_y_ft
            battlebot_vel_ft = round(math.sqrt(battlebot_dx_ft**2 + battlebot_dy_ft**2)/self.dt,2)

            #Pixel Velocity (for display/other conversions) ---------------------------------------------
            prev_x_px = battlebot_data["prev_loc_px"][0]
            prev_y_px = battlebot_data["prev_loc_px"][1]

            curr_x_px = battlebot_data["curr_loc_px"][0]
            curr_y_px = battlebot_data["curr_loc_px"][1]

            #Current and Previous positions
            
            battlebot_dx_px = curr_x_px-prev_x_px
            battlebot_dy_px = curr_y_px-prev_y_px
            battlebot_vel_px = round(math.sqrt(battlebot_dx_px**2 + battlebot_dy_px**2)/self.dt)

            #Calculates theta if the robot is moving(pixel values are probably slightly more accurate)
            if battlebot_vel_px != 0:
                battlebot_theta = math.atan2(battlebot_dy_px,battlebot_dx_px)
                self.battlebot_history[key]["theta"] = battlebot_theta
            else:
                battlebot_theta = battlebot_data["theta"]

            #Storing battlebot info
            battlebot_Kinematics["loc_ft"].append(battlebot_data["curr_loc_ft"])
            battlebot_Kinematics["loc_px"].append(battlebot_data["curr_loc_px"])
            battlebot_Kinematics["theta"].append(battlebot_theta)
            battlebot_Kinematics["vel_px"].append(battlebot_vel_px)
            battlebot_Kinematics["vel_ft"].append(battlebot_vel_ft)

        return battlebot_Kinematics, housebot_Kinematics


    
def draw_velocity(housebot_Kinematics, battlebot_Kinematics, top_view, circle_radius = 10, vel_thresh = 1):
    #Draws a green circle in the center of detected housebots
    center = housebot_Kinematics['loc_px']
    radius = circle_radius
    color = (0,255,0)
    cv.circle(top_view, center,radius, color, -1)

    #Draws a green line showing the velocity of the robot if it is known to be moving
    start_point = center
    vel = housebot_Kinematics["vel_px"]
    theta = housebot_Kinematics['theta']
    end_point = (round(start_point[0] - vel*math.cos(theta)), round(start_point[1] - vel*math.sin(theta)))
    if(vel != 0 and housebot_Kinematics['vel_ft'] > vel_thresh):
        cv.line(top_view,start_point,end_point,color,5)

    for center, vel_px, theta, vel_ft in zip(battlebot_Kinematics['loc_px'],battlebot_Kinematics['vel_px'],battlebot_Kinematics['theta'],battlebot_Kinematics['vel_ft']):
        #Draws a red circle in the center of detected battlebots
        radius = circle_radius
        color = (0,0,255)
        cv.circle(top_view, center,radius, color, -1)

        #Draws a red line showing the velocity of the robot if it is known to be moving
        start_point = center
        vel = vel_px
        end_point = (round(start_point[0] + vel*math.cos(theta)), round(start_point[1] + vel*math.sin(theta)))
        if(vel != 0 and vel_ft > vel_thresh):
            cv.line(top_view,start_point,end_point,color,5)


    return top_view




if __name__ == '__main__':

    model = YOLO("detector.pt")
    cv.namedWindow('image')
    fourcc = cv.VideoWriter_fourcc(*'MJPG')  


    # gets the names of all videos in the Test_Videos folder
    video_names = os.listdir('Test_Videos')
    video_names = [name for name in video_names if name.endswith('.mp4')]

    vid_counter = 0
    pic_counter = 0

    #For removing the background
    fgbg = cv.createBackgroundSubtractorMOG2()
    knn = cv.createBackgroundSubtractorKNN()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

    

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

        #For real time display (Comment out for ML Implementation)
        frame_time = 1000 / fps



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
                time_List = []


             #Transforms the frame to a top down view
            top_view = tI.transform_img(frame,vertices)
            top_view = ut.normalize_nn(top_view)


            #Background removal =======================================================================================
            #KNN subtracktor
            # dynamic_mask = knn.apply(top_view)
            # top_view_subtracted = cv.bitwise_and(top_view,top_view,mask = dynamic_mask)

            #Robot Tracking ==========================================================================================

            dt = 30 #time between tracking in milliseconds 

            # Calculates the scale of the arena in the first frame
            if frames == 0:
                #Scales the arena
                scale, arena_bounds = find_scale()
                #Initializes the velocity detector
                vD = velocity_detector(dt)


            #Checks robot location every dt ms
            if(calc_Time(fps,frames_AI) >= dt or frames == 0):
                start = time.perf_counter() # More precise than time.time()

                #Detects robots
                results = model.track(top_view,conf = 0.3,verbose=False,persist = True)[0]

                #Extracts data from the model
                battlebot_loc_px, housebot_loc_px = extract_loc(results)

                #Processes the model data to find the real location
                battlebot_loc, housebot_loc = process_loc(battlebot_loc_px,housebot_loc_px,scale,arena_bounds)

                #Updates the current position of the robots in the arena
                vD.update_hist(battlebot_loc,housebot_loc)

                #Calculates housebot and battlebot kinematics
                battlebot_Kinematics, housebot_Kinematics = vD.calc_Kinematics()

                
                #     tR.init_data(data_static,scale,arena_bounds,top_view)
                # else:
                #     #Only works after first frame and ONLY if there is a major change in position
                #     tR.update_housebot_kinematics(data_static,dynamic_mask)

                end = time.perf_counter()

                time_List.append(round(end - start,3))
                #Only matters after first frame calculated
                if frames != 0 and calc_Time(fps,frames_AI) >= dt:
                    print(f"Inference Time: {np.mean(time_List)} seconds")
                    time_List = []
                
                frames_AI = 0

             # #Shows results on video
            annotated_frame = draw_velocity(housebot_Kinematics, battlebot_Kinematics, top_view)

            # #Displays the frame
            cv.imshow("Robot Detection", annotated_frame)

            #For real time display (Comment out for ML Implementation)
            elapsed = (time.time() - start) * 1000
            wait = max(1, int(frame_time - elapsed))

                

            


            

            key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            frames += 1
            frames_AI +=1

        


 

    recording = 0