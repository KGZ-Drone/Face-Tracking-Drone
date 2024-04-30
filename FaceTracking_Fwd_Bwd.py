from djitellopy import tello
import cv2

tello = tello.Tello()
tello.connect()
battery_level = tello.get_battery()
print(battery_level)
tello.streamon()

def adjust_tello_position(offset_z):
    """
    Adjusts the position of the tello drone based on the offset values given from the frame
    :param offset_z: Area of the face detection rectangle on the frame
    """
    if not 15000 <= offset_z <= 30000 and offset_z != 0:  # [15000 - 30000]
        if offset_z < 15000:
            # tello.move_forward(20)
            tello.send_rc_control(0, 15, 0, 0)
            print('move_forward', offset_z)
        elif offset_z > 30000:
            # tello.move_back(20)
            tello.send_rc_control(0, -15, 0, 0)
            print('move_back', offset_z)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
frame_read = tello.get_frame_read()

tello.takeoff()
tello.move_up(30)

while True:
    frame = frame_read.frame

    cap = tello.get_video_capture()
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # print('height: ', height, '   width: ', width)

    # Calculate frame center
    center_x = int(width / 2)
    center_y = int(height / 2)

    # Draw the center of the frame
    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 8)

    # Convert frame to grayscale in order to apply the haar cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # 인식된 얼굴 이미지의 크기 초기값
    z_area = 0
    face_center_x = center_x
    face_center_y = center_y

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

        face_center_x = x + int(h / 2)
        face_center_y = y + int(w / 2)
        z_area = w * h
        # print(z_area)

        cv2.circle(frame, (face_center_x, face_center_y), 10, (0, 0, 255), 8)
        cv2.putText(frame, f'[{z_area}]', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)


    cv2.putText(frame, f'[{z_area}]', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)

    # offset uopdate
    adjust_tello_position(z_area)

    cv2.imshow('Tello detection...', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        tello.land()
        break

# Stop the BackgroundFrameRead and land the drone
tello.end()
cv2.destroyAllWindows()
