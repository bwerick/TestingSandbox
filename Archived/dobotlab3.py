from pydobot.dobot import PTPMode as MODE_PTP
from pydobot import Dobot
import pydobot
import time
import cv2
from ultralytics import YOLO
device = pydobot.Dobot(port="/dev/tty.usbmodem21201")  # For Windows, replace COMXX with the actual port, e.g., COM3
cap = cv2.VideoCapture(0)   # Make sure your camera index is correct  
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Current FPS (reported by driver):", cap.get(cv2.CAP_PROP_FPS))
print("Current Resolution: {}x{}".format(
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
))
model = YOLO('/Users/erickduarte/Downloads/yolov8s.pt')  

prev_t = time.time()
win_name = 'Camera Stream + YOLO'

(x, y, z, r, j1, j2, j3, j4) = device.pose()  # Get the current position and joint angles



pose = [x, y, z, r]
joint = [j1, j2, j3, j4] 
print(f"pose: {pose}, j: {joint}")
# position, joint = pose.position, pose.joints
print(pose)
device.speed(80, 80)


allr =0

#home
homex = 209.7 
homey = -3.9
homez = 136.7

#inspect
middlex = 205.1
middley = 11.6
middlez = -30

#pick up
pickupx=300
pickupy=-19.7
pickupz= -57.1

#pallet A
palletAx=285.4
palletAy=107.6
palletAz=-48.9

#pallet B
palletBx=277.3
palletBy= -158.2
palletBz=-48.9

food = ["apple", "banana", "pizza"]
item = ""
# Preferred way to move
#start position
device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False)
while True:
    cycletimestart = time.time()
    #move to inspect position
    device.move_to(x=middlex, y=middley, z=middlez, r=allr, wait=False)
    time.sleep(2)

    #inspect item type
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        results = model(frame, verbose=False)  
        r = results[0]
        

        if r.boxes is not None and len(r.boxes) > 0:

            xyxy = r.boxes.xyxy.cpu().numpy()

            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names  

            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2  
                label = f"{names[k]} {c:.2f}"
                if c >= 0.79:
                    print(names[k])
                    item = names[k]
                    print(c)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                
        now = time.time()
        fps = 1.0 / (now - prev_t)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win_name, frame)
        if c >= 0.79: #cv2.waitKey(1) & 0xFF == ord('q')
            break

    cap.release()
    cv2.destroyAllWindows()


    #move to pick up
    device.move_to(x=pickupx, y=pickupy, z=pickupz, r=allr, wait=False)#move to pick up position
    time.sleep(2)
    device.suck(True)#suck item
    device.move_to(x=middlex, y=middley, z=middlez, r=allr, wait=False)#move to inspect position
    time.sleep(2)

    if item in food:
        #move to pallet A
        device.move_to(x=palletAx, y=palletAy, z=palletAz, r=allr, wait=False)#move to pallet A
        time.sleep(2)
        device.suck(False)#drop item
    else:
        #move to pallet B
        device.move_to(x=palletBx, y=palletBy, z=palletBz, r=allr, wait=False)#move to pallet B
        time.sleep(2)
        device.suck(False)#drop item
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#move back to home
device.move_to(x=homex, y=homey, z=homez, r=allr, wait=False) #move to home position
# Calculate cycle time
cycletimeend = time.time()
cycletime = cycletimeend - cycletimestart
print(f"Cycle time: {cycletime:.2f} seconds")
device.close()  # Close the connection to the robot
