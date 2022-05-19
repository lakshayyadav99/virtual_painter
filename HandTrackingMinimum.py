import cv2
import mediapipe as mp #made by google
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #this function helps in drawing lines on hand


#variables that will help us in displaying FPS
pTime = 0 #previous time
cTime = 0 #curr time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converting to rgb
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:  #we are extracting info whether there's multi hand or single
            for id,lm in enumerate(handlms.landmark): #id is basically a number that tells the about part of hand;
                h,w,c = img.shape                     #Like id=0 is for lower palm position;
                cx,cy = int(lm.x*w),int(lm.y*h) #position of parts of finger and hand

                # if id==0: #this will make a circle around lower part of plam
                #     cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)


            mpDraw.draw_landmarks(img,handlms, mpHands.HAND_CONNECTIONS) # here we are drawing lines on hand in webcam


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #putting fps on image
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3);

    cv2.imshow("Image", img)
    cv2.waitKey(1)