from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations

import pyttsx3
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

speak("starting")    
print("starting")
print(time.time())

start_time = time.time()
# Distance constants 
KNOWN_DISTANCE = 50 #INCHES
PERSON_WIDTH = 3000 #INCHES



# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
RED = (255,0,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv2.FONT_HERSHEY_COMPLEX

# class_names = []
# with open("classes.txt", "r") as f:
#     class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net


def is_close(p1, p2):
    """
    # 1. Calculate Euclidean Distance between two points
    :param:
    p1, p2 = two points for calculating Euclidean Distance
    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1**2 + p2**2)
    return dst 

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

 

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, 10)
 


def convertBack(x, y, w, h): 
    """
    # 2. Converts center coordinates to rectangle coordinates     
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax



def cvBoxes(detections, img):
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet
    :return:
    img with bounding box
    """

    # 3. Filtering the person class from detections and get bounding box centroid for each person detection

    if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0
        data_list =[]
        # print("======")								# We inialize a variable called ObjectId and set it to 0
        curr_time = time.time()
        print(curr_time-start_time)
        for label, confidence, bbox in detections:				# In this if statement, we filter all the detections for persons only
            # Check for the only person name tag 
            name_tag = label   # Coco file has string of all the names
            # print(name_tag)
            # print(bbox)
            # speak(name_tag)
            if name_tag == 'person':
                pass;    

            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]      	# Store the center points of the detections
            xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox            
            # Append center point of bbox for persons detected.
            centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox
            
            color= COLORS[int(objectId) % len(COLORS)]
    
            distance = distance_finder(focal_person, PERSON_WIDTH, xmax-xmin)

            x = xmin
            y = ymin-2

            if distance <= 2:

                speak(name_tag)
                cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (255, 0, 0), 2) 

                cv2.putText(img, name_tag, (xmin,ymin-14), FONTS, 0.5, RED, 2)

                cv2.rectangle(img, (x, y-3), (x+150, y+23),BLACK,-1 )
                cv2.putText(img, f'In {round(distance,0)} ft', (x+5,y+13), FONTS, 0.48, RED, 2)
                # print(centroid_dict[objectId])

            else:
                cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (0, 255, 0), 2)

                cv2.putText(img, name_tag, (xmin,ymin-14), FONTS, 0.5, color, 2)

                # cv2.rectangle(img, (x, y-3), (x+150, y+23),BLACK,-1 )
                # cv2.putText(img, f'In { round(distance,0)} ft', (x+5,y+13), FONTS, 0.48, GREEN, 2)



            objectId += 1 #Increment the index for each detection 
        



        red_zone_list = [] # List containing which Object id is in under threshold distance condition. 
        red_line_list = []



        
        # for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
        #     dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	# Check the difference between centroid x: 0, y :1
        #     distance = is_close(dx, dy) 			# Calculates the Euclidean distance
        #     if distance < 75.0:						# Set our social distance threshold - If they meet this condition then..
        #         if id1 not in red_zone_list:
        #             red_zone_list.append(id1)       #  Add Id to a list
        #             red_line_list.append(p1[0:2])   #  Add points to the list
        #         if id2 not in red_zone_list:
        #             red_zone_list.append(id2)		# Same for the second id 
        #             red_line_list.append(p2[0:2])
        
        # for idx, bbox in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
        #     print(bbox)
        #     if idx in red_zone_list:   # if id is in red zone list
        #         cv2.rectangle(img, (bbox[2], bbox[3]), (bbox[4], bbox[5]), (255, 0, 0), 2) # Create Red bounding boxes  #starting point, ending point size of 2
        #     else:
        #         cv2.rectangle(img, (bbox[2], bbox[3]), (bbox[4], bbox[5]), (0, 255, 0), 2) # Create Green bounding boxes


    	# 3. Display risk analytics and risk indicators
    
        text = "No of Objects: %s" % str(len(detections)) 			# Count People at Risk
        location = (10,25)												# Set the location of the displayed text
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Display Text

        for check in range(0, len(red_line_list)-1):					# Draw line between nearby bboxes iterate through redlist items
            start_point = red_line_list[check] 
            end_point = red_line_list[check+1]
            check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
            check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
            if (check_line_x < 75) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed. 
        #=================================================================#
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    # if netMain is None:
    #     netMain = darknet.load_net_custom(configPath.encode(
    #         "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    # if metaMain is None:
    #     metaMain = darknet.load_meta(metaPath.encode("ascii"))

    network, class_names, class_colors = darknet.load_network(configPath,  metaPath, weightPath, batch_size=1)

    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(0)
    # Use this if you want to connect your webcam
    # cap = cv2.VideoCapture("./Input/final2.mp4")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 1, frame_width // 1
    # print("Video Reolution: ",(width, height))

    out = cv2.VideoWriter(
            "./test2_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
            (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)
        image = cvBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print("Video Write Completed...")

if __name__ == "__main__":
    YOLO()