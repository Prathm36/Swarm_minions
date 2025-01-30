import numpy as np
import cv2
import cv2.aruco as aruco
# import RPi.GPIO as GPIO
import time

def ball_detection():

    cap = cv2.VideoCapture(0)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grayy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(grayy)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        # S_low = max(50, int(mean_s * 0.5))
        # V_low = max(50, int(mean_v * 0.5))

        lower = np.array([0, 120, 15])
        upper = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])



        v_eq = cv2.equalizeHist(v)
        hsv_equalized = cv2.merge([h, s, v_eq])

        mask = cv2.inRange(hsv_equalized, lower, upper)
        mask2 = cv2.inRange(hsv_equalized, lower2, upper2)
        kernel = np.ones((5, 5), "uint8")
        mask = cv2.dilate(mask, kernel)
        mask2 = cv2.dilate(mask2, kernel)
        mask = mask + mask2

        result = cv2.bitwise_and(hsv, hsv, mask=mask)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(gray, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=25, maxRadius=150)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        if circles is None:
            yield((640, 480))
        try:
            # if circles and ids.any() is not None:
            circles = np.uint16(np.around(circles))
            # Find the circle with the largest radius
            largest_circle = max(circles[0, :], key=lambda c: c[2])
            aruco.drawDetectedMarkers(frame, corners, ids)
            max_close = 0
            max_close_l = []
            # Draw the largest circle
            cv2.circle(frame, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 255, 0), 2)
            cv2.circle(frame, (largest_circle[0], largest_circle[1]), 2, (0, 255, 0), 3)
        except:
            continue


        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            for pic, contour, j, marker_id in zip(enumerate(contours), enumerate(ids)):
                l_temp = []
                area = cv2.contourArea(contour)
                x, y = int(corners[j][0][0][0]), int(corners[j][0][0][1])
                cv2.putText(frame, f"ID: {marker_id[0]}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                l_temp = [marker_id[0], (corners[j][0][0][0] + corners[j][0][2][0]) / 2,
                          (corners[j][0][0][1] + corners[j][0][2][1]) / 2]
                if max_close < l_temp[1]:
                    max_close = l_temp[1]
                    max_close_l = l_temp
        except:
            continue

        yield (int(largest_circle[0]), int(largest_circle[1]), max_close_l)

        if (area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        cv2.imshow("image", result)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


for coords in ball_detection():
    print(f"Circle detected at: {coords}")

'''def goal_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        S_low = max(50, int(mean_s * 0.5))
        V_low = max(50, int(mean_v * 0.5))

        lower = np.array([100, 120, 50])
        upper = np.array([130, 255, 255])
        ''''''lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])''''''

        v_eq = cv2.equalizeHist(v)
        hsv_equalized = cv2.merge([h, s, v_eq])

        mask = cv2.inRange(hsv_equalized, lower, upper)
        #mask2 = cv2.inRange(hsv_equalized, lower2, upper2)
        kernel = np.ones((5, 5), "uint8")
        mask = cv2.dilate(mask, kernel)
        #mask2 = cv2.dilate(mask2, kernel)
        #mask = mask + mask2

        result = cv2.bitwise_and(hsv, hsv, mask=mask)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        flag = 0

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 300):
                flag = 1
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                yield((x+w)/2, (y+h)/2)
                cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        if flag == 0:
            yield((640, 480))
        cv2.imshow("image", result)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

for coords in goal_detection():
    print(f"Circle detected at: {coords}")'''

def goal_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mean_s = np.mean(s)
        mean_v = np.mean(v)
        S_low = max(50, int(mean_s * 0.5))
        V_low = max(50, int(mean_v * 0.5))

        lower = np.array([100, 120, 50])
        upper = np.array([130, 255, 255])

        v_eq = cv2.equalizeHist(v)
        hsv_equalized = cv2.merge([h, s, v_eq])

        mask = cv2.inRange(hsv_equalized, lower, upper)
        kernel = np.ones((5, 5), "uint8")
        mask = cv2.dilate(mask, kernel)

        result = cv2.bitwise_and(hsv, hsv, mask=mask)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        flag = False
        largest_contour = None
        largest_area = 0

        # Find the largest contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300 and area > largest_area:
                largest_area = area
                largest_contour = contour
                flag = True  # Set flag to True if a valid contour is detected

        # If a largest contour is found, draw its rectangle and yield its center coordinates
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            imageFrame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            yield ((x + w) / 2, (y + h) / 2)

        if not flag:
            yield (640, 480)

        cv2.imshow("image", result)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

'''for coords in goal_detection():
    print(f"Circle detected at: {coords}")'''



'''def motor_control(in1, in2, en, x):

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(en, GPIO.OUT)

    # Initialize PWM for speed control
    pwm = GPIO.PWM(en, 1000)  # Frequency of 1 kHz
    pwm.start(50)  # Start with 50% duty cycle for medium speed

    try:
        if x == 'f':
            print("Moving forward")
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif x == 'b':
            print("Moving backward")
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            print("Invalid command. Use 'f' for forward or 'b' for backward.")

    finally:
        # Optional: Clean up resources when done
        pwm.stop()
        GPIO.cleanup()'''

def motor_control(in1, in2, en, direction):
    '''GPIO.setmode(GPIO.BCM)
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(en, GPIO.OUT)'''

    '''pwm = GPIO.PWM(en, 1000)  # Frequency: 1 kHz
    pwm.start(50)  # Start with 50% duty cycle'''

    if direction == 'f':  # Move forward
        # GPIO.output(in1, GPIO.HIGH)
        # GPIO.output(in2, GPIO.LOW)
        print("f")
    elif direction == 'b':  # Move backward
        # GPIO.output(in1, GPIO.LOW)
        # GPIO.output(in2, GPIO.HIGH)
        print("b")
    else:  # Stop motor if invalid direction
        # GPIO.output(in1, GPIO.LOW)
        # GPIO.output(in2, GPIO.LOW)
        print("else")
        print("Invalid command. Stopping motor.")

# Cleanup should only happen at the end of the program.
# pwm.stop() and GPIO.cleanup() can be called when shutting down.

# Example usage
# control_motor(24, 23, 25, 'f')  # Move forward
# control_motor(24, 23, 25, 'b')  # Move backward

def detect_aruco_from_cam():
    """
    Detects ArUco markers in real-time using a webcam and displays their IDs.
    Press 'q' to exit the webcam feed.
    """
    # Open the webcam (0 for default camera)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load the predefined ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()

    # Start video capture
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert frame to grayscale
        grayy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(grayy)
        #print(corners)
        l_aruco = []

        # Draw detected markers and display IDs
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            max_close = 0
            max_close_l = []
            for i, marker_id in enumerate(ids):
                l_temp = []
                x, y = int(corners[i][0][0][0]), int(corners[i][0][0][1])
                cv2.putText(frame, f"ID: {marker_id[0]}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                l_temp = [marker_id[0], (corners[i][0][0][0]+corners[i][0][2][0])/2, (corners[i][0][0][1]+corners[i][0][2][1])/2]
                if max_close < l_temp[1]:
                    max_close = l_temp[1]
                    max_close_l = l_temp
            yield(max_close_l)
        # Show the output
        cv2.imshow("ArUco Marker Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the function
#detect_aruco_from_cam()


'''in11 = 2
in12 = 3
in21 = 4
in22 = 5
en1 = 6
en2 = 7
#sensor = 0

ball_detection()

for a in ball_detection():

    if a[0] > 300 and a[0] < 340:
        motor_control(in11, in21, en1, 's')
        break
    else:
        motor_control(in11, in21, en1, 'f')'''

'''init = time.time()

while True:
    motor_control(in11, in21, en1, 'f')
    motor_control(in21, in22, en2, 'f')
    if int(time.time()-init) > 20:
        motor_control(in11, in21, en1, 's')
        motor_control(in21, in22, en2, 's')
        break

# for a in goal_detection():
#     a = next(goal_detection())
#     if a[0] > 300 and a[0] < 340:
#         motor_control(in11, in21, en1, 's')
#         break
#     else:
#         motor_control(in11, in21, en1, 'f')

init = time.time()
while True:
    motor_control(in11, in21, en1, 'f')
    motor_control(in21, in22, en2, 'f')
    if int(time.time()-init) > 20:
        motor_control(in11, in21, en1, 's')
        motor_control(in21, in22, en2, 's')
        break'''




