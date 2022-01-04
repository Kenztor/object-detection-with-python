from math import radians
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
import random
import imutils
from pynput.keyboard import  Controller
 
keyboard = Controller()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#ตั้งค่าจุด Mark (x,y) บนจอแสดงผล เช่น x_enemy, y_enemy = x, y (ซึ่ง scale ของ x,y ขึ้นอยู่กับตั้งค่าขนาดของภาพด้วย(การกำหนดfx=, fy=))) #ไปตั้งค่าก่อนที่บรรทัดที่ 37
x_enemy, y_enemy= 650, 400
a_enemy, b_enemy= 200, 300
c_enemy, d_enemy= 400, 150
e_enemy, f_enemy= 400, 400

def enemy():
  
  cv2.circle(image, (x_enemy,y_enemy), 25, (0, 0, 200), 3)
  cv2.circle(image, (a_enemy,b_enemy), 25, (255, 100, 0), 3)
  cv2.circle(image, (c_enemy,d_enemy), 25, (150, 255, 50), 3)
  cv2.circle(image, (e_enemy,f_enemy), 25, (71, 223, 246), 3)
 
video = cv2.VideoCapture(0)
 
with mp_hands.Hands(max_num_hands=8,min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while video.isOpened():
        
        _, frame = video.read()
           
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image = cv2.resize(image,None,fx=1.25, fy=1.25) #ตั้งค่าของขนาดภาพ เช่น 1080*720 fx=2.25, fy=1.5
        imageHeight, imageWidth, _ = image.shape
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
        enemy()
 
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)
 
 
        if results.multi_hand_landmarks != None:
          for handLandmarks in results.multi_hand_landmarks:
            for point in mp_hands.HandLandmark:
 
    
                normalizedLandmark = handLandmarks.landmark[point]
                pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
    
                point=str(point)
                if point=='HandLandmark.INDEX_FINGER_TIP':
                  
                  try:
                     
                     cv2.circle(image, (pixelCoordinatesLandmark[0], pixelCoordinatesLandmark[1]), 20, (255, 102, 255), -1)
                    
                     if ((x_enemy-25<pixelCoordinatesLandmark[0]<x_enemy or x_enemy<pixelCoordinatesLandmark[0]<x_enemy+25) 
                        and (y_enemy-25<pixelCoordinatesLandmark[1]<y_enemy or y_enemy<pixelCoordinatesLandmark[1]<y_enemy+25)):
                        keyboard.press('a')
                        keyboard.release('a')
                        print("found_1")
                        
                        
                     elif ((a_enemy-25<pixelCoordinatesLandmark[0]<a_enemy or a_enemy<pixelCoordinatesLandmark[0]<a_enemy+25) 
                        and (b_enemy-25<pixelCoordinatesLandmark[1]<b_enemy or b_enemy<pixelCoordinatesLandmark[1]<b_enemy+25)):
                            keyboard.press('b')
                            keyboard.release('b')
                            print("found_2")
                            

                     elif ((c_enemy-25<pixelCoordinatesLandmark[0]<c_enemy or c_enemy<pixelCoordinatesLandmark[0]<c_enemy+25) 
                        and (d_enemy-25<pixelCoordinatesLandmark[1]<d_enemy or d_enemy<pixelCoordinatesLandmark[1]<d_enemy+25)):
                            keyboard.press('c')
                            keyboard.release('c')
                            print("found_3")


                     elif ((e_enemy-25<pixelCoordinatesLandmark[0]<e_enemy or e_enemy<pixelCoordinatesLandmark[0]<e_enemy+25) 
                        and (f_enemy-25<pixelCoordinatesLandmark[1]<f_enemy or f_enemy<pixelCoordinatesLandmark[1]<f_enemy+25)):
                            keyboard.press('d')
                            keyboard.release('d')
                            print("found_4")                         
                  except:
                    pass
                enemy()
                        
        cv2.imshow('Hand Tracking', image)
 
        if cv2.waitKey(10) & 0xFF == ord('q'): #กด q เพื่อหยุดการทำงาน
            break
 
video.release()
cv2.destroyAllWindows()