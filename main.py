import cv2
import numpy as np
# open cv real time object detection

def calculate_distance(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    center1 = (x1 + w1 / 2, y1 + h1 / 2)
    center2 = (x2 + w2 / 2, y2 + h2 / 2)
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    return distance

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFiles = 'coco.names'
with open(classFiles, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize object tracking
tracked_objects = {}

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    
    # Update tracked objects
    for idx, box in enumerate(bbox):
        box_id = f'{classIds[idx]}_{tuple(box)}'  # Unique identifier for each bounding box
        if all(calculate_distance(box, tracked_box) > 20 for tracked_box in tracked_objects.values()):
            tracked_objects[box_id] = box  # Add new object to tracked_objects
            cv2.rectangle(img, tuple(box), color=(0, 255, 0), thickness=2)
    
    # Display object count
    object_count = len(tracked_objects)
    cv2.putText(img, f'Objects: {object_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
