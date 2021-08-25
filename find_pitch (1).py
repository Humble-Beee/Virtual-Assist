# from pitch_detection import *
import pyttsx3
import speech_recognition as sr 
import requests
import json
import moviepy.editor as mp
import datetime

import os
import glob
import time
import librosa
import librosa.display
import scipy
import numpy as np

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

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
    my_clip = mp.VideoFileClip(r"videoplayback.mp4")
    my_clip.audio.write_audiofile(r"my_result.wav")
    file="C:\\Users\\Rishu raj\\OneDrive\\Desktop\\HWI\\my_result.wav"
    y, sr = librosa.load(file)
    # Run pitch detection
    pitch = Pitch_Detect(y, sr=8000, Ws=512)
    alert=pitch.run()
    print(alert)
    speak(alert)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

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

def condition():
    while(True):
        print('listening....')
        query=input("command sir--")#takeCommand().lower
        if 'check traffic' in query:
            check_traffic()
        elif 'go offline'in query:
            speak("Thank you..., sir closing all systems")
            quit()
            break
        elif 'time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            speak(f"Sir, the time is {strTime}")
            print(f"Sir, the time is {strTime}")
        elif 'date and time' in query:
            dt()
        elif 'weather report' in query:
            speak("wait sir telling weather report...")
            weather()


def main():
    speak("hello sir, Have a good day.")  
    print("hello sir, Have a good day.")
    condition()

    

if __name__ == "__main__":
    main()