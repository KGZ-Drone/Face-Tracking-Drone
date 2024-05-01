from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2, math, time
from djitellopy import Tello

tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

model = YOLO('yolov8n.pt')
# model = YOLO('Mask_best_200_epoch.pt')

while True:
    cv2.waitKey(1)
    tello_video_image = frame_read.frame
    results: list[Results] = model(tello_video_image)
    annotated_frame = results[0].plot()
    # boxes = results[0].boxes  # Boxes object for bbox outputs

    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
exit()