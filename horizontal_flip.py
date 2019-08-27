import cv2
import os.path
import os
import numpy as np
import csv
import sys


DATA_DIR = "DATA/3F"


for filename in os.listdir(DATA_DIR):
    print int(filename[0])
    img = cv2.imread(os.path.join(DATA_DIR, filename))
    flipped = cv2.flip(img,1)
    cv2.imwrite(os.path.join(DATA_DIR, filename), flipped)