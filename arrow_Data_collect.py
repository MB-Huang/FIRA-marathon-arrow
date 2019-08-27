import cv2
import os.path
import random
cam = cv2.VideoCapture(1)

cv2.namedWindow("test")
counter=0
img_counter = random.randint(0,999999999)
writefile=0
label=-1
while True:
    ret, frame = cam.read()

    if not ret:
        break
    k = cv2.waitKey(100)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 >= 48 and k%256<=51:
        if k%256-48==label:    
            writefile=1-writefile
        label=k%256-48
        # SPACE pressed
    if writefile:
        img_name = "DATA/"+str(label)+"/{}.png".format(img_counter)#0 left 1 straight 2 right  3 None 
        if not os.path.isfile(img_name): 
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            print(counter)
        else:
            print("{} exist!".format(img_name))
        counter += 1
        img_counter = random.randint(0,999999999)
    if counter ==50:
        writefile=False
        counter=0
    
    x=100 
    y=20 
    w=440 
    h=440
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
    cv2.imshow("test", frame)
cam.release()

cv2.destroyAllWindows()