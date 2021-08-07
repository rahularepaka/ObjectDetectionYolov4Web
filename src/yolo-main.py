import cv2
import numpy as np

cap = cv2.VideoCapture(0)

width = 320
height = 320

#Threshold Values
Conf_threshold = 0.4
NMS_threshold = 0.4

#Colours
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# empty list
class_name = []

# for reading all the datasets from the coco.names file into the array
with open("models\\coco.names", 'rt') as f:
    class_name = f.read().rstrip('\n').split('\n')

# configration and weights file location
model_config_file = "models\\yolov4-tiny.cfg"
model_weight = "models\\yolov4-tiny.weights"

# darknet files
net = cv2.dnn.readNet(model_weight, model_config_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Load Model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def detect():
    classes, scores, boxes = model.detect(
            frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s : %f" % (class_name[classid[0]], score)
        
        cv2.rectangle(frame, box, color, 1)
        cv2.putText(frame, label, (box[0], box[1]-10),
                   cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
        

while True:
    ret, frame = cap.read()
    
    detect()

    cv2.imshow('frame', frame)

    if cv2.waitKey(10) &  0xFF == ord('a'):
        break
    

cap.release()
cv2.destroyAllWindows()
