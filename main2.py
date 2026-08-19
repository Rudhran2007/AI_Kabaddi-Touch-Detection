import cv2
from ultralytics import YOLO
import math
model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("svideo.mp4")
RAIDER_ID = 1
touched_players = set()
HAND_TOUCH_DIST = 25
CONFIRM_FRAMES = 3
prev_hands = []
touch_counter = {}
def point_to_box_distance(px, py, box):
    x1, y1, x2, y2 = box
    cx = max(x1, min(px, x2))
    cy = max(y1, min(py, y2))
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
def is_moving_towards(prev, curr, target_box):
    if prev is None:
        return False
    prev_dist = point_to_box_distance(prev[0], prev[1], target_box)
    curr_dist = point_to_box_distance(curr[0], curr[1], target_box)
    return curr_dist < prev_dist
while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    if results[0].boxes.id is None:
        continue
    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    keypoints = results[0].keypoints.xy.cpu().numpy()
    raider_hands = []
    for box, pid, kp in zip(boxes, ids, keypoints):
        if pid == RAIDER_ID:
            raider_hands = [kp[9], kp[10]]
            break
    if len(prev_hands) != len(raider_hands):
        prev_hands = raider_hands
    for box, pid in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        if pid != RAIDER_ID:
            for i, (hx, hy) in enumerate(raider_hands):
                if hx == 0 and hy == 0:
                    continue
                dist = point_to_box_distance(hx, hy, box)
                moving = is_moving_towards(prev_hands[i], (hx, hy), box)

                if dist < HAND_TOUCH_DIST and moving:
                    if pid not in touch_counter:
                        touch_counter[pid] = 0

                    touch_counter[pid] += 1

                    if touch_counter[pid] >= CONFIRM_FRAMES:
                        touched_players.add(pid)
                else:
                    touch_counter[pid] = 0

        if pid in touched_players:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {pid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    prev_hands = raider_hands
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 95), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    cv2.putText(
        frame,
        f"Touched Count: {len(touched_players)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    touched_list = sorted(touched_players)

    if len(touched_list) == 0:
        lines = ["Touched IDs: None"]
    else:
        chunk_size = 4
        lines = []
        for i in range(0, len(touched_list), chunk_size):
            chunk = touched_list[i:i + chunk_size]
            line = "Touched IDs: " + ", ".join(map(str, chunk))
            lines.append(line)

    y = 65
    for line in lines[:2]:  
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )
        y += 25

    cv2.imshow("Kabaddi Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()