import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from vidgear.gears import CamGear
import winsound  # Import the library for playing sound (Windows)

# Function to play an alarm sound
def play_alarm():
    duration = 1000  # milliseconds
    frequency = 440  # Hz (adjust as needed)
    winsound.Beep(frequency, duration)

stream = CamGear(source='https://www.youtube.com/watch?v=gFRtAAmiFbE', stream_mode=True, logging=True).start() # YouTube Video URL as input
count = 0
threshold = 5  # Set the threshold for the number of objects detected

while True:
    frame = stream.read()
    count += 1
    if count % 10 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    bbox, label, conf = cv.detect_common_objects(frame)
    frame = draw_bbox(frame, bbox, label, conf)
    num_objects = len(label)

    if num_objects > threshold:
        print(f"Alarm: {num_objects} objects detected!")
        play_alarm()  # Play the alarm sound

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.stop()
cv2.destroyAllWindows()
