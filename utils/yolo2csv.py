import argparse
import csv
from pathlib import Path

import cv2



# define class
CLASS = {'0': "fresh", '1': "old"}

def convertCoord(yolo_format, img_h, img_w):
    info = yolo_format.split()
    
    class_name = CLASS[info[0]]  # class id
    x_center, y_center = float(info[1]), float(info[2])
    w, h = float(info[3]), float(info[4])

    x1, y1 = int((x_center-w/2)*img_w), int((y_center-h/2)*img_h)
    x3, y3 = int((x_center+w/2)*img_w), int((y_center+h/2)*img_h)

    return [x1, y1, x3, y3], class_name

def yolo2csv(image_dir, label_dir):
    img_path = Path(image_dir)
    label_path = Path(label_dir)
    label_list = [y.stem for y in label_path.iterdir()]

    annotations = []
    for x in img_path.iterdir():
        img = cv2.imread(str(x))
        heigh, weight = img.shape[0], img.shape[1]
        
        if x.stem in label_list:
            with open(label_path / (x.stem+'.txt'), 'r') as f:
                for line in f.readlines():
                    coord, class_name = convertCoord(line, heigh, weight)
                    objects = [str(x)]
                    objects.extend(coord)
                    objects.append(class_name)
                    annotations.append(objects)
        else:
            annotations.append([str(x), "", "", "", "", ""])

    with open("annotations.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(annotations)

    with open("class.csv", 'w', newline='') as f:
        reCLASS = [[v, k] for k, v in CLASS.items()]

        writer = csv.writer(f)
        writer.writerows(reCLASS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default='dataset/images', help="image dir path")
    parser.add_argument("--labels", type=str, default='dataset/labels', help="yolo labels dir path")
    opt = parser.parse_args()
    print(opt)

    yolo2csv(image_dir=opt.images, label_dir=opt.labels)