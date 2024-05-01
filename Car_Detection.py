import cv2 as cv
import cvzone
from ultralytics import YOLO
import math

cap = cv.VideoCapture(r"Videos\cars.mp4")
# cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
counter = 0
while True:
    _, frame = cap.read()
    result = model(frame,stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2,y2 = box.xyxy[0]
            x1,y1, x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # cv.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            w,h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            
            conf =  math.ceil(box.conf[0]*100)/100
            cls = box.cls[0]
            cls = int(cls)
            if classNames[cls]=='car' or classNames[cls]=="truck" or classNames[cls]=="bus":
                cvzone.putTextRect(frame,f"{classNames[cls]} {conf}",(max(0,x1),max(35,y1)),scale=0.8,thickness=1)
                cvzone.cornerRect(frame,(x1,y1,w,h), l=9,t=1)
                counter+=1
                print(counter)
    cv.imshow("Cool",frame)
    cv.waitKey(1)