import cv2
import cvlib as cv
from vidgear.gears import CamGear
import numpy as np

# Function to create a density map with moving crosses and dynamic boxes
def create_density_map(image, bboxes, num_people, alarm_threshold, min_people_per_box):
    density_map = np.zeros_like(image, dtype=np.uint8)
    background_color = (173, 216, 230)  # Set the background color (white)

    # Fill the density map with the background color
    density_map[:] = background_color

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) // 2, (y1 + y2) // 2  # Use the center of the bounding box

        # Draw horizontal crossbar
        cv2.line(density_map, (x - 10, y), (x + 10, y), (0, 0, 255), 2)

        # Draw vertical crossbar
        cv2.line(density_map, (x, y - 10), (x, y + 10), (0, 0, 255), 2)

    # Initialize the count of boxes
    num_boxes = 0

    # Sort bounding boxes by their x-coordinate (left to right)
    bboxes.sort(key=lambda x: (x[0] + x[2]) / 2)

    # Detect groups of at least min_people_per_box people who are closeby
    group = []
    for bbox in bboxes:
        if not group or bbox[0] - group[-1][2] <= 20:
            group.append(bbox)
        else:
            if len(group) >= min_people_per_box:
                # Draw a border around the group
                x1 = group[0][0]
                y1 = min(bbox[1] for bbox in group)
                x2 = group[-1][2]
                y2 = max(bbox[3] for bbox in group)
                border_color = (0, 0, 255)  # Red color for the border
                cv2.rectangle(density_map, (x1, y1), (x2, y2), border_color, 2)
                num_boxes += 1
            group = [bbox]

    # Display the count of people at the bottom left corner
    count_message = f"People Count: {num_people}"
    cv2.putText(density_map, count_message, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if the alarm threshold is exceeded and display the alarm message
    if num_people > alarm_threshold:
        alarm_message = f"ALARM: {num_people} people detected in {num_boxes} groups!"
        cv2.putText(density_map, alarm_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return density_map, num_boxes

# Threshold for the number of people to trigger an alarm
people_threshold = 5

stream = CamGear(source='https://www.youtube.com/watch?v=gFRtAAmiFbE', stream_mode=True, logging=True).start()
count = 0

while True:
    frame = stream.read()
    count += 1
    if count % 10 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    bbox, label, conf = cv.detect_common_objects(frame)

    # Filter the results to include only "person" labels
    person_indices = [i for i, lbl in enumerate(label) if lbl == "person"]
    filtered_bbox = [bbox[i] for i in person_indices]

    # Count the number of people
    num_people = len(filtered_bbox)

    # Create a density map with moving crosses, dynamic boxes, and people count
    density_map, num_boxes = create_density_map(frame, filtered_bbox, num_people, people_threshold, min_people_per_box=5)

    # Display the density map with the moving crosses, dynamic boxes, and people count
    cv2.imshow("Density Map (People)", density_map)

    if num_people > people_threshold:
        print(f"ALARM: {num_people} people detected in {num_boxes} groups!")

    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.stop()
cv2.destroyAllWindows()
