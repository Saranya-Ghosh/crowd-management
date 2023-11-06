import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from vidgear.gears import CamGear
import numpy as np

# Function to create a density map
def create_density_map(image, bboxes):
    height, width = image.shape[:2]
    density_map = np.zeros((height, width), dtype=np.float32)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        density_map[y1:y2, x1:x2] += 1

    return density_map

stream = CamGear(source='https://www.youtube.com/watch?v=gFRtAAmiFbE', stream_mode=True, logging=True).start()
count = 0

while True:
    frame = stream.read()
    count += 1
    if count % 10 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    bbox, label, conf = cv.detect_common_objects(frame)
    frame = draw_bbox(frame, bbox, label, conf)

    # Count the number of people in the frame
    person_count = sum(1 for l in label if l == 'person')

    # Create a density map based on the detected people
    density_map = create_density_map(frame, bbox)

    # Normalize the density map
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the density map to a color map for visualization
    density_map_color = cv2.applyColorMap(np.uint8(density_map), cv2.COLORMAP_JET)

    # Display the density map and person count
    cv2.putText(frame, f'Person Count: {person_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Density Map", density_map_color)

    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.stop()
cv2.destroyAllWindows()
