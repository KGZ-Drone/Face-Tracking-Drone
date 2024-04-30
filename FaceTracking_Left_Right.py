from djitellopy import tello
import cv2

tello = tello.Tello()
tello.connect()
battery_level = tello.get_battery()
print(battery_level)
tello.streamon()

def adjust_tello_position(offset_x):
    """
    Adjusts the position of the tello drone based on the offset values given from the frame
    :param offset_x: Offset between center and face x coordinates
    """
    if not -90 <= offset_x <= 90 and offset_x != 0:
        # offset_x가 음수이면 시계 반대 방향으로 일정 거리 만큼 이동
        if offset_x < 0:
            print('rotate_counter_clockwise')
            # tello.rotate_counter_clockwise(10)
            tello.send_rc_control(0, 0, 0, -15)

        # offset_x가 양수이면 시계 방향으로 일정 거리 만큼 이동
        elif offset_x > 0:
            print('rotate_clockwise')
            # tello.rotate_clockwise(10)
            tello.send_rc_control(0, 0, 0, 15)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
frame_read = tello.get_frame_read()

tello.takeoff()
tello.move_up(30)

while True:
    frame = frame_read.frame

    cap = tello.get_video_capture()
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print('height: ', height, '   width: ', width)

    # Calculate frame center
    center_x = int(width / 2)
    center_y = int(height / 2)

    # Draw the center of the frame
    cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 6)

    # Convert frame to grayscale in order to apply the haar cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)

    # 얼굴을 탐지하지 못한 경우 [No Face] 출력
    # if len(faces) == 0:
    #     print('face not found')
    #     cv2.putText(frame, '[No Face]', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)

    # If a face is recognized, draw a rectangle over it and add it to the face list
    face_center_x = center_x
    face_center_y = center_y

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

        face_center_x = x + int(w / 2)
        face_center_y = y + int(h / 2)

        cv2.circle(frame, (face_center_x, face_center_y), 10, (0, 0, 255), 6)

    # Calculate recognized face offset from center
    offset_x = face_center_x - center_x

    cv2.putText(frame, f'[{offset_x}]', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)
    adjust_tello_position(offset_x)

    # Display the resulting frame
    cv2.imshow('Tello detection...', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        tello.land()
        break

# Stop the BackgroundFrameRead and land the drone
tello.end()
cv2.destroyAllWindows()