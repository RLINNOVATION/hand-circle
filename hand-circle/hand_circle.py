import cv2
import mediapipe as mp
import math

# Initialize Mediapipe and OpenCV objects
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to calculate distance between two landmarks
def calculate_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2) * 1000  # Scale appropriately

# Capture video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks for visual reference
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get specific landmarks: thumb_tip and index_finger_tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Calculate distance between thumb and index finger tips
            diameter = calculate_distance(thumb_tip, index_tip)
            
            # Count fingers (a basic check using y-coordinates to determine if fingers are up)
            fingers_up = [hand_landmarks.landmark[i].y < hand_landmarks.landmark[i-2].y for i in 
                          [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                           mp_hands.HandLandmark.MIDDLE_FINGER_TIP]]
            
            # Draw based on finger count
            if fingers_up.count(True) == 2:  # Two fingers up, draw a circle
                cv2.circle(frame, 
                           (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])), 
                           int(diameter), (0, 255, 0), 2)
            elif fingers_up.count(True) == 1:  # One finger up, draw a line
                cv2.line(frame, 
                         (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])), 
                         (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])), 
                         (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
