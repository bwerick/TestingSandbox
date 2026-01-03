from ultralytics import YOLO
import cv2

model = YOLO("/Users/erickduarte/git/TestingSandbox/runs/ttt_xo_mixed/weights/best.pt")
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    res = model.predict(frame, conf=0.5)
    cv2.imshow("XO", res[0].plot())
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
