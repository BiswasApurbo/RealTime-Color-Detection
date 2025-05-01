import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Color database with realistic RGB
color_labels = {
    "Red": (220, 20, 60),
    "Green": (0, 200, 64),
    "Blue": (30, 144, 255),
    "Yellow": (255, 255, 0),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Gray": (128, 128, 128),
    "Orange": (255, 140, 0),
    "Brown": (150, 75, 0),
    "Pink": (255, 182, 193),
    "Purple": (128, 0, 128)
}

# Maximum possible RGB distance
MAX_RGB_DIST = np.sqrt(255**2 + 255**2 + 255**2)

def get_color_name_and_confidence(r, g, b):
    min_dist = float("inf")
    closest_color = "Other"

    for name, (cr, cg, cb) in color_labels.items():
        dist = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if dist < min_dist:
            min_dist = dist
            closest_color = name

    confidence = max(0, 100 - (min_dist / MAX_RGB_DIST) * 100)  # Scale confidence
    if min_dist > 120:  # Optional strict threshold
        return "Other", 0
    return closest_color, int(confidence)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb tip (4) and index fingertip (8)
            x1 = int(hand_landmarks.landmark[4].x * w)
            y1 = int(hand_landmarks.landmark[4].y * h)
            x2 = int(hand_landmarks.landmark[8].x * w)
            y2 = int(hand_landmarks.landmark[8].y * h)

            # Midpoint between thumb and index finger
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            roi_size = 15
            x_start, y_start = max(cx - roi_size, 0), max(cy - roi_size, 0)
            x_end, y_end = min(cx + roi_size, w), min(cy + roi_size, h)

            roi = img[y_start:y_end, x_start:x_end]
            if roi.size > 0:
                avg_color = np.mean(roi.reshape(-1, 3), axis=0).astype(int)
                b, g, r = avg_color
                color_name, confidence = get_color_name_and_confidence(r, g, b)

                # Draw color box and info
                cv2.rectangle(img, (10, 10), (220, 60), (int(b), int(g), int(r)), -1)
                label_text = f"{color_name} ({confidence}%)"
                text_color = (0, 0, 0) if confidence > 50 else (255, 255, 255)
                cv2.putText(img, label_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

                # Show rectangle on ROI
                cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)

    cv2.imshow("Color Detection with Confidence", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
