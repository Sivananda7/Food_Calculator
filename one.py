from ultralytics import YOLO
import cv2
import yaml

img = "C:/Users/kaavi/OneDrive/Desktop/food/train/images/8_jpg.rf.4ef02d51c89b2c273fedec34286c58d6.jpg"
imgAnot = "C:/Users/kaavi/OneDrive/Desktop/food/train/labels/8_jpg.rf.4ef02d51c89b2c273fedec34286c58d6.txt"

data_yaml_file = "C:/Users/kaavi/OneDrive/Desktop/food/data.yaml"

with open(data_yaml_file, 'r') as file:
    data = yaml.safe_load(file)

label_names = data['names']
print(label_names)

img = cv2.imread(img)
H,W,_ = img.shape

with open(imgAnot,'r') as file:
    lines = file.readlines()

annotations = []
for line in lines:
    values = line.split()
    label = values[0]

    x,y,w,h = map(float, values[1:])
    annotations.append((label,x,y,w,h))
print(annotations)

for annotations in annotations:
    label, x,y,w,h = annotations
    label_name = label_names[int(label)]


    x1 = int((x - w / 2 ) * W)
    y1 = int((y - h / 2 ) * H)
    x2 = int((x + w / 2 ) * W)
    y2 = int((y + h / 2 ) * H)

    cv2.rectangle(img, (x1,y1),(x2,y2),(200,200,0),1)

    cv2.putText(img, label_name,(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (200,200,0),2)

cv2.imwrite("img.jpg", img)