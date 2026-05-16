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
                self.battlebot_history[key]["omega"] = 0
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
                self.housebot_history["omega"] = 0
            except:
                self.housebot_history["prev_loc_ft"] = (-10,-10)
                self.housebot_history["prev_loc_px"] = (-10,-10)
                self.housebot_history["curr_loc_ft"] = (-10,-10)
                self.housebot_history["curr_loc_px"] = (-10,-10)
                self.housebot_history["theta"] = -10 
                self.housebot_history["omega"] = 0
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
            vel_thresh: The velocity threshhold for detection in pix/s, set to 10 px/s by default
        Outputs:
            battlebot_Kinematics: A list of data containing the following information
                loc_px: (x,y) tuple of the location in x,y coordinates in pixels
                loc_ft: (x,y) tuple of the location in x,y coordinates in feet
                theta: The robot's current direction of travel in rad
                omega: The robot's current angular velocity in rad/s
                vel_px: The robot's velocity in pixels/s (for display)
                vel_ft: The robot's velocity in ft/s
                id: The robot's current ID

            housebot_Kinematics: A dictionary containing the following data about the housebot
                loc_px: (x,y) tuple of the location in x,y coordinates in pixels
                loc_ft: (x,y) tuple of the location in x,y coordinates in feet
                theta: The robot's current direction of travel
                omega: The robot's current angular velocity
                vel_px: The robot's velocity in pixels/s (for display)
                vel_ft: The robot's velocity in ft/s
    """
    def calc_Kinematics(self, vel_thresh = 10):
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
        if housebot_vel_px > vel_thresh:
            housebot_theta = math.atan2(housebot_dy_px,housebot_dx_px)

            #Calculates omega if it is relevant
            if self.housebot_history["theta"] != -10:
                housebot_omega = housebot_theta - self.housebot_history["theta"]
            else:
                housebot_omega = 0 
            self.housebot_history["theta"] = housebot_theta
            self.housebot_history["omega"] = housebot_omega
            
        else:
            housebot_theta = self.housebot_history["theta"]
            housebot_omega = 0
        
        housebot_omega = housebot_omega/self.dt

        #Storing housebot info
        housebot_Kinematics = {}
        housebot_Kinematics["loc_ft"] = self.housebot_history["curr_loc_ft"]
        housebot_Kinematics["loc_px"] = self.housebot_history["curr_loc_px"]
        housebot_Kinematics["theta"] = housebot_theta
        housebot_Kinematics["vel_px"] = housebot_vel_px
        housebot_Kinematics["vel_ft"] = housebot_vel_ft
        housebot_Kinematics["omega"] = housebot_omega


     

        # Calculate velocity and orientation for the detected battlebots

        battlebot_Kinematics = {}
        battlebot_Kinematics["loc_ft"] = []
        battlebot_Kinematics["loc_px"] = []
        battlebot_Kinematics["theta"] = []
        battlebot_Kinematics["vel_px"] = []
        battlebot_Kinematics["vel_ft"] = []
        battlebot_Kinematics["id"] = []
        battlebot_Kinematics["omega"] = []

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
            if battlebot_vel_px > vel_thresh:
                battlebot_theta = math.atan2(battlebot_dy_px,battlebot_dx_px)

                 #Calculates omega if it is relevant
                if self.battlebot_history[key]["theta"] != -10:
                    battlebot_omega = battlebot_theta - self.battlebot_history[key]["theta"]
                else:
                    battlebot_omega = 0 

                self.battlebot_history[key]["theta"] = battlebot_theta
                self.battlebot_history[key]["omega"] = battlebot_omega
            else:
                battlebot_theta = battlebot_data["theta"]
                battlebot_omega = 0
            
            battlebot_omega = battlebot_omega/self.dt

            #Storing battlebot info
            battlebot_Kinematics["loc_ft"].append(battlebot_data["curr_loc_ft"])
            battlebot_Kinematics["loc_px"].append(battlebot_data["curr_loc_px"])
            battlebot_Kinematics["theta"].append(battlebot_theta)
            battlebot_Kinematics["vel_px"].append(battlebot_vel_px)
            battlebot_Kinematics["vel_ft"].append(battlebot_vel_ft)
            battlebot_Kinematics["id"].append(key)
            battlebot_Kinematics["omega"].append(battlebot_omega)

        return battlebot_Kinematics, housebot_Kinematics


"""
running_average_filter: A running average filter
    Storage:
        n: the number of data points stored for the averaging filter (5 by default)
        dt: The time differential between frames in seconds
        battlebot_history:
            id: A list of the detected battlebot IDs
            omega: A list of lists which correspond to the detected battlebots' omega
            theta: A list of lists which correspond to the detected battlebots' theta
            vel_px: A list of lists which correspond to the velocities of the detected battlebots in px/s
            vel_ft: A list of lists which correspond to the velocities of the detected battlebots in ft/s
            loc_px: A list of lists which correspond to the location of the detected battlebots in px
            loc_ft: A list of lists which correspond to the location of the detected battlebots in ft
        housebot_history:
            omega: A list which corresponds to the detected housebot's omega
            theta: A list which corresponds to the detected housebot's theta
            vel_px: A list which corresponds to the velocity of the detected housebot in px/s
            vel_ft: A list which corresponds to the velocity of the detected housebot in ft/s
            loc_px: A list which corresponds to the location of the detected housebot in px
            loc_ft: A list which corresponds to the location of the detected housebot in ft
    Functions:
        init_filter: Initializes the running average filter
        upadte_battlebots: Removes battlebots from the data when they are not detected and adds the newly detected IDs
        update_history: Updates the history
        get_averages: Gets the data from the running average filter
"""
class running_average_filter:
    def __init__(self,dt,n=5):
        self.n = n
        self.dt = dt/1000
        self.battlebot_history = {}
        self.housebot_history = {}

    
    """
    init_filter: Initializes the running average filter
        Inputs:
            housebot_Kinematics: The current kinematics of the housebot
            battlebot_Kinematics: The current kinematics of the battlebots
        Output:
            None
    """
    def init_filter(self,housebot_Kinematics,battlebot_Kinematics):
        try:
            #Initializes the housebot
            self.housebot_history["omega"] = [0] #Assumed to be still when starting
            self.housebot_history["theta"] = [-10] #Current orientation assumed to be unknown
            self.housebot_history["vel_px"] = [0] # Assumed to be still when starting
            self.housebot_history["vel_ft"] = [0] #Assumed to be still when starting
            self.housebot_history["loc_px"] = [housebot_Kinematics["loc_px"]] #known location
            self.housebot_history["loc_ft"] = [housebot_Kinematics["loc_ft"]] #known location
        except:
            #Initializes the housebot
            self.housebot_history["omega"] = [0] #Assumed to be still when starting
            self.housebot_history["theta"] = [-10] #Current orientation assumed to be unknown
            self.housebot_history["vel_px"] = [0] # Assumed to be still when starting
            self.housebot_history["vel_ft"] = [0] #Assumed to be still when starting
            self.housebot_history["loc_px"] = [(-10,-10)] #unknown location
            self.housebot_history["loc_ft"] = [(-10,-10)] #unknown location

        #Initializes the battlebots (ALL LISTS)
        self.battlebot_history["id"] = battlebot_Kinematics["id"]
        self.battlebot_history["omega"] = []
        self.battlebot_history["theta"] = []
        self.battlebot_history["vel_px"] = []
        self.battlebot_history["vel_ft"] = []
        self.battlebot_history["loc_px"] = []
        self.battlebot_history["loc_ft"] = []

        #Adds the data for each robot
        for loc_px, loc_ft in zip(battlebot_Kinematics["loc_px"], battlebot_Kinematics["loc_ft"]):
            self.battlebot_history["omega"].append([0]) #Assumed to be still
            self.battlebot_history["theta"].append([-10]) #Currently unknown
            self.battlebot_history["vel_px"].append([0]) # Assumed to be still
            self.battlebot_history["vel_ft"].append([0]) # Assumed to be still
            self.battlebot_history["loc_px"].append([loc_px])
            self.battlebot_history["loc_ft"].append([loc_ft])

    """
    update_battlebots: Removes battlebots from the data when they are not detected and adds any newly detected ones
        Inputs:
            battlebot_Kinematics: A list of the ids of the detected battlebots
        Output: 
            None
    """        
    def update_battlebots(self,battlebot_Kinematics):
        detected_ids = battlebot_Kinematics["id"] #detected Robot IDs

        #Removes Undetected Battlebots ======================================
        loc = 0
        historical_ids = self.battlebot_history["id"]
        for hist_id in historical_ids:
            #If the historical id is not in the detected ids
            if hist_id not in detected_ids: 
                self.battlebot_history["id"].pop(loc)
                self.battlebot_history["omega"].pop(loc)
                self.battlebot_history["theta"].pop(loc)
                self.battlebot_history["vel_px"].pop(loc)
                self.battlebot_history["vel_ft"].pop(loc)
                self.battlebot_history["loc_px"].pop(loc)
                self.battlebot_history["loc_ft"].pop(loc)
            loc+=1
        
        #Adds any new battlebots ====================================
        loc = 0
        for detected_id in detected_ids:
            #If it is not in the historical ids
            if detected_id not in historical_ids:
                self.battlebot_history["id"].append(detected_id)
                self.battlebot_history["omega"].append([0])
                self.battlebot_history["theta"].append([-10])
                self.battlebot_history["vel_px"].append([0])
                self.battlebot_history["vel_ft"].append([0])
                self.battlebot_history["loc_px"].append([battlebot_Kinematics["loc_px"][loc]])
                self.battlebot_history["loc_ft"].append([battlebot_Kinematics["loc_ft"][loc]])
            loc+=1

        
    """
    update_history: Adds new relevant data to the history to help run the filter
        Input: 
            battlebot_Kinematics: The kinematic data for each battlebot
            housebot_Kinematics: The kinematic data for each housebot
        Output:
            None
    """
    def update_history(self,battlebot_Kinematics,housebot_Kinematics):
        # Updates housebot data ======================================================
        self.housebot_history["omega"].append(housebot_Kinematics["omega"])
        self.housebot_history["theta"].append(housebot_Kinematics["theta"])
        self.housebot_history["vel_px"].append(housebot_Kinematics["vel_px"])
        self.housebot_history["vel_ft"].append(housebot_Kinematics["vel_ft"])
        self.housebot_history["loc_px"].append(housebot_Kinematics["loc_px"])
        self.housebot_history["loc_ft"].append(housebot_Kinematics["loc_ft"])

        #Removes any extraneous data from the housebot_history ===================================
        if len(self.housebot_history["omega"]) > self.n:
            self.housebot_history["omega"].pop(0)
            self.housebot_history["theta"].pop(0)
            self.housebot_history["vel_px"].pop(0)
            self.housebot_history["vel_ft"].pop(0)
            self.housebot_history["loc_px"].pop(0)
            self.housebot_history["loc_ft"].pop(0)

        #Updates battlebot data ====================================================
        loc_detect = 0
        for detected_id in battlebot_Kinematics["id"]:
            loc_hist = 0
            for hist_id in self.battlebot_history["id"]:
                if hist_id == detected_id:
                    self.battlebot_history["omega"][loc_hist].append(battlebot_Kinematics["omega"][loc_detect])
                    self.battlebot_history["theta"][loc_hist].append(battlebot_Kinematics["theta"][loc_detect])
                    self.battlebot_history["vel_px"][loc_hist].append(battlebot_Kinematics["vel_px"][loc_detect])
                    self.battlebot_history["vel_ft"][loc_hist].append(battlebot_Kinematics["vel_ft"][loc_detect])
                    self.battlebot_history["loc_px"][loc_hist].append(battlebot_Kinematics["loc_px"][loc_detect])
                    self.battlebot_history["loc_ft"][loc_hist].append(battlebot_Kinematics["loc_ft"][loc_detect])
                loc_hist += 1
            loc_detect += 1

        #Removes any extraneous data from battlebot_history ================================================
        for loc in range(0,len(self.battlebot_history["id"])):
            if(len(self.battlebot_history["omega"][loc]) > self.n):
                self.battlebot_history["omega"][loc].pop(0)
                self.battlebot_history["theta"][loc].pop(0)
                self.battlebot_history["vel_px"][loc].pop(0)
                self.battlebot_history["vel_ft"][loc].pop(0)
                self.battlebot_history["loc_px"][loc].pop(0)
                self.battlebot_history["loc_ft"][loc].pop(0)
        
    """
    get_averages: gets the average kinematics for the battlebots
        Input: 
            None
        Output:
            battlebot_Kinematics: The average kinematic data for each battlebot
            housebot_Kinematics: The average kinematic data for the housebot
    """
    def get_averages(self):
        #Calculates housebot_Kinematics ===============================================
        #Removes invalid data i.e. anything with -10 values
        housebot_omega = round(np.mean(self.housebot_history["omega"]).item(),3)

        #Removes invalid data for theta
        housebot_theta = []
        for t in self.housebot_history["theta"]:
            if t != -10:
                housebot_theta.append(self.housebot_history["theta"])
        if(len(housebot_theta)==0):
            housebot_theta = -10
        else:
            housebot_theta = round(np.mean(housebot_theta).item(),3)
        
        housebot_vel_px = round(np.mean(self.housebot_history["vel_px"]).item())
        housebot_vel_ft = round(np.mean(self.housebot_history["vel_ft"]).item(),3)

        #Pixel location ================================
        housebot_loc_px_x = []
        housebot_loc_px_y = []

        for loc in self.housebot_history["loc_px"]:
            x = loc[0]
            y = loc[1]

            if(x != -10 and y != -10):
                housebot_loc_px_x.append(x)
                housebot_loc_px_y.append(y)

        if(len(housebot_loc_px_x) == 0):
            housebot_loc_px = (-10,-10)
        else:
            housebot_loc_px_x = round(np.mean(housebot_loc_px_x).item())
            housebot_loc_px_y = round(np.mean(housebot_loc_px_y).item())
            housebot_loc_px = (housebot_loc_px_x,housebot_loc_px_y)

        #Real location ================================
        housebot_loc_ft_x = []
        housebot_loc_ft_y = []

        for loc in self.housebot_history["loc_ft"]:
            x = loc[0]
            y = loc[1]

            if(x != -10 and y != -10):
                housebot_loc_ft_x.append(x)
                housebot_loc_ft_y.append(y)

        if(len(housebot_loc_ft_x) == 0):
            housebot_loc_ft = (-10,-10)
        else:
            housebot_loc_ft_x = round(np.mean(housebot_loc_ft_x).item(),3)
            housebot_loc_ft_y = round(np.mean(housebot_loc_ft_y).item(),3)
            housebot_loc_ft = (housebot_loc_ft_x,housebot_loc_ft_y)


        #Adds all the housebot data
        housebot_Kinematics = {}
        housebot_Kinematics["omega"] = housebot_omega
        housebot_Kinematics["theta"] = housebot_theta
        housebot_Kinematics["vel_px"] = housebot_vel_px
        housebot_Kinematics["vel_ft"] = housebot_vel_ft
        housebot_Kinematics["loc_px"] = housebot_loc_px
        housebot_Kinematics["loc_ft"] = housebot_loc_ft

        #Calculates battlebot Kinematics
        battlebot_Kinematics = {}
        battlebot_Kinematics["id"] = self.battlebot_history["id"]
        #Calculates omega
        battlebot_Kinematics["omega"] = []
        for omega_data in self.battlebot_history["omega"]:
            omega = round(np.mean(omega_data).item(),3)
            battlebot_Kinematics["omega"].append(omega)
        #Calculates theta
        battlebot_Kinematics["theta"] = []
        for theta_data in self.battlebot_history["theta"]:
            #Removes invalid data for theta
            theta = []
            for t in theta_data:
                if t != -10:
                    theta.append(theta_data)
            if(len(theta)==0):
                theta = -10
            else:
                theta = round(np.mean(theta).item(),3)
            battlebot_Kinematics["theta"].append(theta)
        #Calculates vel_px
        battlebot_Kinematics["vel_px"] = []
        for vel_px_data in self.battlebot_history["vel_px"]:
            vel_px = round(np.mean(vel_px_data).item())
            battlebot_Kinematics["vel_px"].append(vel_px)
        #Calculates vel_ft
        battlebot_Kinematics["vel_ft"] = []
        for vel_ft_data in self.battlebot_history["vel_ft"]:
            vel_ft = round(np.mean(vel_ft_data).item(),3)
            battlebot_Kinematics["vel_ft"].append(vel_ft)
        #Calculates loc_px
        battlebot_Kinematics["loc_px"] = []
        for loc_px_data in self.battlebot_history["loc_px"]:
            loc_px_x = []
            loc_px_y = []

            for loc in loc_px_data:
                x = loc[0]
                y = loc[1]

                if(x != -10 and y != -10):
                    loc_px_x.append(x)
                    loc_px_y.append(y)

            if(len(loc_px_x) == 0):
                loc_px = (-10,-10)
            else:
                loc_px_x = round(np.mean(loc_px_x).item())
                loc_px_y = round(np.mean(loc_px_y).item())
                loc_px = (loc_px_x,loc_px_y)
            battlebot_Kinematics["loc_px"].append(loc_px)
        #Calculates loc_ft
        battlebot_Kinematics["loc_ft"] = []
        for loc_ft_data in self.battlebot_history["loc_ft"]:
            loc_ft_x = []
            loc_ft_y = []

            for loc in loc_ft_data:
                x = loc[0]
                y = loc[1]

                if(x != -10 and y != -10):
                    loc_ft_x.append(x)
                    loc_ft_y.append(y)

            if(len(loc_ft_x) == 0):
                loc_ft = (-10,-10)
            else:
                loc_ft_x = round(np.mean(loc_ft_x).item(),3)
                loc_ft_y = round(np.mean(loc_ft_y).item(),3)
                loc_ft = (loc_ft_x,loc_ft_y)
            battlebot_Kinematics["loc_ft"].append(loc_ft)

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
    end_point = (round(start_point[0] + vel*math.cos(theta)), round(start_point[1] - vel*math.sin(theta)))
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
            frame_start = time.perf_counter()  # Start timing the whole frame
            
            ret, frame = cap.read()
            if not ret:
                print(f"Can't receive frame (stream end?). Exiting ...")
                break

            frame = ut.normalize_img(frame)

            if(frames == 0):
                tI = ut.transform_img(frame)
                detectVertices = tI.detect_Vertices()
                vertices = detectVertices[0]
                time_List = []
                last_inference_time = frame_start  # Initialize inference timer

            top_view = tI.transform_img(frame,vertices)
            top_view = ut.normalize_nn(top_view)

            dt = 30

            if frames == 0:
                scale, arena_bounds = find_scale()
                vD = velocity_detector(dt)
                run_filter = running_average_filter(dt)

            if(calc_Time(fps,frames_AI) >= dt or frames == 0):
                # Calculate actual dt from last inference
                now = time.perf_counter()
                if frames != 0:
                    actual_dt_ms = (now - last_inference_time) * 1000
                    vD.dt = actual_dt_ms / 1000  # Update before kinematics
                    run_filter.dt = actual_dt_ms / 1000  # Update before kinematics
                last_inference_time = now

                results = model.track(top_view,conf = 0.3,verbose=False,persist = True,max_det=7)[0]

                battlebot_loc_px, housebot_loc_px = extract_loc(results)
                battlebot_loc, housebot_loc = process_loc(battlebot_loc_px,housebot_loc_px,scale,arena_bounds)
                vD.update_hist(battlebot_loc,housebot_loc)
                battlebot_Kinematics, housebot_Kinematics = vD.calc_Kinematics()

                #Running Average filter initialization and running
                if(frames == 0):
                    run_filter.init_filter(housebot_Kinematics,battlebot_Kinematics)
                else:
                    run_filter.update_battlebots(battlebot_Kinematics)
                    run_filter.update_history(battlebot_Kinematics,housebot_Kinematics)
                    battlebot_Kinematics, housebot_Kinematics = run_filter.get_averages()


                


                inference_time = time.perf_counter() - now
                time_List.append(round(inference_time, 3))

                if frames != 0 and calc_Time(fps,frames_AI) >= dt:
                    time_List = []

                frames_AI = 0

            annotated_frame = draw_velocity(housebot_Kinematics, battlebot_Kinematics, top_view)
            cv.imshow("Robot Detection", annotated_frame)

            # Consistent timing using perf_counter throughout
            elapsed = (time.perf_counter() - frame_start) * 1000
            wait = max(1, int(frame_time - elapsed))

            key = cv.waitKey(wait) & 0xFF  # Also fixed: was waitKey(1), now uses calculated wait

            if key == ord('q'):
                break
            frames += 1
            frames_AI += 1


 

    recording = 0