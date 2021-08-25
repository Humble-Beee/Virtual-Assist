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
import speech_recognition as sr 
import requests
import json
import moviepy.editor as mp
import datetime 
import glob 
import librosa
import librosa.display
import scipy 


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

async def speak1(audio):
    engine.say(audio)
    engine.runAndWait()

start_time = time.time()
# Distance constants 
KNOWN_DISTANCE = 50 #INCHES
PERSON_WIDTH = 3000 #INCHES

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
RED = (255,0,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv2.FONT_HERSHEY_COMPLEX



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
        # print(curr_time-start_time)
        for label, confidence, bbox in detections:				# In this if statement, we filter all the detections for persons only
            # Check for the only person name tag 
            name_tag = label   # Coco file has string of all the names
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

                print( f'{name_tag} In {round(distance,0)} ft')
                speak(name_tag)
                cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (255, 0, 0), 2) 

                cv2.putText(img, name_tag, (xmin,ymin-14), FONTS, 0.5, RED, 2)

                cv2.rectangle(img, (x, y-3), (x+150, y+23),BLACK,-1 )
                cv2.putText(img, f'In {round(distance,0)} ft', (x+5,y+13), FONTS, 0.48, RED, 2)
                # print(centroid_dict[objectId])

            else:
                cv2.rectangle(img, (xmin,ymin), (xmax, ymax), (0, 255, 0), 2)

                cv2.putText(img, name_tag, (xmin,ymin-14), FONTS, 0.5, color, 2)

                cv2.rectangle(img, (x, y-3), (x+150, y+23),BLACK,-1 )
                cv2.putText(img, f'In { round(distance,0)} ft', (x+5,y+13), FONTS, 0.48, GREEN, 2)



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
    # cap = cv2.VideoCapture("./Input/kolkata.mp4")

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
    yolo_start_time = time.time()
    while True:


        if time.time()-yolo_start_time > 120:
            cv2.destroyAllWindows()
            break

        key_pressed = cv2.waitKey(10)

        if key_pressed == ord('q'):
            cv2.destroyAllWindows()
            break

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
        cv2.waitKey(1)
        out.write(image)


    cap.release()
    out.release()
    print("Video Write Completed...")


# from pitch_detection import *
# engine = pyttsx3.init('sapi5')
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)

#....
class Pitch_Detect:

    def __init__(self, y, sr, Ws):
        self.y = y
        self.sr = sr
        self.N = Ws


    def hpf(self, filter_stop_freq=50, filter_pass_freq=200, filter_order=1001):
        nyquist_rate = self.sr / 2.
        desired = (0, 0, 1, 1)
        bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
        filter_coefs = scipy.signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

        filtered_audio = scipy.signal.filtfilt(filter_coefs, [1], self.y)
        return filtered_audio


    def camdf(self, y_clip, tau):
        D = 0.0
        for n in range(self.N):
            D += abs(y_clip[(n + tau)%self.N] - y_clip[n])
        return D


    def pitch_curve(self):
        l = self.N
        N = len(self.y[:l+1])
        pitch_list = []
        ran = (len(self.y)//(l//2))-2
        
        for i in range(ran):
            camdf_list = []
            y_clip = self.y[(l//2)*i:(l//2)*i+l+1]
            for i in range(l):
                camdf_list.append(self.camdf(y_clip=y_clip, tau=i))
            interval = camdf_list[4:100]
            min_D = min(interval)
            pitch_detected = round(self.sr/(interval.index(min_D)+4),2)
            pitch_list.append(pitch_detected)
        
        return pitch_list, ran


    def run(self):
        y = self.hpf(filter_stop_freq=50, filter_pass_freq=200, filter_order=1001)
        pitch_list, ran = self.pitch_curve()
        max_pitch=max(pitch_list)
        # print(max_pitch)
        if 400<max_pitch<3000:
            return "Attention!,Car is approaching."
        else:
            return "free to  move."    


def check_traffic():
    my_clip = mp.VideoFileClip(r"./Input/videoplayback.mp4")
    my_clip.audio.write_audiofile(r"./Input/my_result.wav")
    file="./Input/my_result.wav"
    y, sr = librosa.load(file)
    # Run pitch detection
    pitch = Pitch_Detect(y, sr=8000, Ws=512)
    alert=pitch.run()
    print(alert)
    speak(alert)
 

def dt():
    strTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
    print(f"Sir, the date and time is {strTime}")
    speak(f"Sir, the date and time is {strTime}")

def weather():
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
    CITY = "New Delhi"
    API_KEY = "94b83c8e37003458a17dccc3c28e74fb"
    URL = BASE_URL + "q=" + CITY + "&appid=" + API_KEY    
    response = requests.get(URL)

    if response.status_code == 200:
        data = response.json()
        main = data['main']
        temperature = main['temp']
        temp_feel_like = main['feels_like']  
        humidity = main['humidity']
        pressure = main['pressure']
        weather_report = data['weather']
        wind_report = data['wind']
        
        print(" Temperature (in kelvin unit) = " +
                        str(temperature) +  
                "\n atmospheric pressure (in hPa unit) = " +
                        str(pressure) +
                "\n humidity (in percentage) = " +
                        str(humidity) +
                "\n wind speed = " +
                        str(wind_report['speed'])) 
        speak(" Temperature (in kelvin unit) = " +
                        str(temperature) + 
                " atmospheric pressure (in hPa unit) = " +
                        str(pressure) +
                " humidity (in percentage) = " +
                        str(humidity) +
                " wind speed = " +
                        str(wind_report['speed']))  
    else:
        print("Error in the HTTP request")
        speak("there is some error")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        speak("listening ...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except :

        print("Say that again please...")  
        query= "None"    
    return query    



def detect(file):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(file)
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    # lower_yellow = np.array([15,100,100])
    # upper_yellow = np.array([35,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)

    size = img.shape
    # print size

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=10, minRadius=0, maxRadius=30)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=5, minRadius=0, maxRadius=30)

    # traffic light detect
    r = 5
    bound = 4.0 / 10
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                #cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                return "RED"

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                #cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                return "GREEN"

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                return "YELLOW"
                #cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    else:
        return "No traffic signal"

    cv2.imshow('detected results', cimg)
    #cv2.imwrite(path+'//result//'+file, cimg)
    # cv2.imshow('maskr', maskr)
    # cv2.imshow('maskg', maskg)
    # cv2.imshow('masky', masky)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



image_height, image_width = 64, 64
max_images_per_class = 8000

dataset_directory = "UCF50"
classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]

model_output_size = len(classes_list)




# import pafy
# import youtube_dl

from keras.models import load_model
from collections import deque
from moviepy.editor import *

def predict_on_live_video(video_file_path, output_file_path, window_size):
    model = load_model('./akshat.h5')
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))

    starting_time = time.time()
    last_speak = ""
    while True: 

        if time.time()-starting_time > 120:
            cv2.destroyAllWindows()
            break

        key_pressed = cv2.waitKey(10)
        if key_pressed == ord('q'):
            cv2.destroyAllWindows()
            break

        # Reading The Frame
        status, frame = video_reader.read() 

        if not status:
            break

        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]
            # print(predicted_class_name)
            if predicted_class_name != last_speak:
                speak(predicted_class_name)
                last_speak = predicted_class_name
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #print(predicted_class_name)

        video_writer.write(frame)


        cv2.imshow('Predicted Frames', frame)



    # cv2.destroyAllWindows()

    
    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them. 
    video_reader.release()
    video_writer.release()

import speech_recognition as sr
# Initialize the recognizer
r = sr.Recognizer()
# Function to convert text to
# speech

if __name__ == "__main__": 

    # while(True): 
    # query=input("command sir--")#takeCommand().lower
    speak("Virtual Assist By Team Key board Crackers")  
    while(1):
        
        try:
            with sr.Microphone() as source2:

                print("Listening...")
                speak("command sir")
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=0.2)

                #listens for the user's input
                audio2 = r.listen(source2)

                # Using ggogle to recognize audio
                MyText = r.recognize_google(audio2)
                query = MyText.lower()

                print("Did you say "+MyText)
                # SpeakText(MyText)t(MyText)
            
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            speak("unknown error occured please give command on text")
            query=input("text command sir--")#takeCommand().lower

   

        if 'check traffic' in query:
            check_traffic()
        elif 'offline'in query:
            print("Thank you..., sir closing all systems")
            speak("Thank you..., sir closing all systems")
            quit()
            # break
        elif 'time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            speak(f"Sir, the time is {strTime}")
            print(f"Sir, the time is {strTime}")
        elif 'date and time' in query:
            dt()
        elif 'weather' in query:
            print("wait sir telling weather report...")
            speak("wait sir telling weather report...")
            weather()

        elif 'traffic light' in query:

            videoCaptureObject = cv2.VideoCapture(0)
            result = True
            while(result):
                ret,frame = videoCaptureObject.read()
                cv2.imwrite("NewPicture.jpg",frame)
                result = False
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            ans = detect("NewPicture.jpg")

            if ans == "GREEN" or ans == "YELLOW":
                speak("Please wait the signal is live")
            else:
                speak("You can go now")
        
        elif 'distance' in query:
            YOLO()

        elif 'activity' in query:
            # Setting the Window Size which will be used by the Rolling Average Process
            window_size = 25

            output_video_file_path = f'./Input/activity_output.mp4'

            predict_on_live_video("./Input/activity.mp4", output_video_file_path, window_size)

            # VideoFileClip(output_video_file_path).ipython_display(width = 700) 

 

