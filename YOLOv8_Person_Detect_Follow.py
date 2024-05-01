import djitellopy
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2, math, time
import numpy as np
from djitellopy import Tello

# fov = 82.6 / 180 * math.pi
# halftanfov = math.tan(fov / 2)
SKYBLUE = (255, 255, 0)
YELLOW = (255, 0, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

model = YOLO('YOLOv8n.pt')

# init tello
tello: Tello = djitellopy.Tello()
tello.connect()
tello.streamon()

# tello.set_speed(50)
print("Battery is " + str(tello.get_battery()))

# tello.takeoff()
# tello.move_up(40)
# tello.send_rc_control(0, 0, 40, 0)

frame_read = tello.get_frame_read()

while True:
    cv2.waitKey(1)

    tello_video_image = frame_read.frame
    # resized_frame = cv2.resize(rgb_frame, (360, 240))

    results: list[Results] = model(tello_video_image)
    # results: list[Results] = model(resized_frame, )
    # print('results', results)

    # annotated_frame = results[0].plot()
    boxes = results[0].boxes  # Boxes object for bbox outputs

    for box in boxes:
        print(model.names.pop(box.cls.item()))
        if model.names.pop(box.cls.item()) == 'person':
            x1, y1, x2, y2 = box.xyxy.numpy()[0][0].item(), box.xyxy.numpy()[0][1].item(), box.xyxy.numpy()[0][
             2].item(), box.xyxy.numpy()[0][3].item()
            confidence = box.conf.item()
            # print(confidence)
            cv2.putText(tello_video_image, model.names.pop(box.cls.item()) + ' ' + str(round(confidence, 2)) + '%', (int(x1), int(y1)-6), cv2.FONT_ITALIC, 1, GREEN, 2)
            cv2.rectangle(tello_video_image, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 3)

            centerX = int(x1 + (abs(x1 - x2) / 2))
            centerY = int(y1 + (abs(y1 - y2) / 2))
            print('center =', centerX, centerY)

            cv2.circle(tello_video_image, (centerX, centerY), 15, RED, 5)

            # perform tello movements
            if centerX < 170:
                print('yaw -40')
                # tello.send_rc_control(0, 0, 0, -30)
            elif centerX > 450:
                print('yaw 430')
                # tello.send_rc_control(0, 0, 0, 30)
            else:
                print('forward 40')
                # tello.send_rc_control(0, 40, 0, 0)
                # tello.send_rc_control(0, 0, 0, 0)
            break

        # we don't see anything
        # do a scan
        # print('yaw +10')
        # tello.send_rc_control(0, 0, 0, 10)

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", tello_video_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tello.streamoff()
cv2.destroyAllWindows()
exit()