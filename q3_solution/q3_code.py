import cv2
import mediapipe as mp
import time
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=3,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
prev_time = 0

# Landmarks for eyes
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 153, 144, 145]   # subset around left eye
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 380, 373, 374] # subset around right eye

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 720))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = frame.shape

    # ---- Face detection ----
    face_results = face_detection.process(rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 255), 2)

    # ---- Face landmarks (nose + accurate eye centers) ----
    mesh_results = face_mesh.process(rgb)
    if mesh_results.multi_face_landmarks:
        for landmarks in mesh_results.multi_face_landmarks:
            # Nose tip
            nose = landmarks.landmark[1]
            cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 0), -1)

            # Left eye center = mean of selected landmarks
            left_eye_pts = np.array([(landmarks.landmark[i].x * w,
                                       landmarks.landmark[i].y * h) for i in LEFT_EYE_IDX])
            left_eye_center = np.mean(left_eye_pts, axis=0).astype(int)
            cv2.circle(frame, tuple(left_eye_center), 6, (255, 0, 0), -1)

            # Right eye center
            right_eye_pts = np.array([(landmarks.landmark[i].x * w,
                                        landmarks.landmark[i].y * h) for i in RIGHT_EYE_IDX])
            right_eye_center = np.mean(right_eye_pts, axis=0).astype(int)
            cv2.circle(frame, tuple(right_eye_center), 6, (255, 0, 0), -1)

    # ---- FPS ----
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Detection + Nose & Accurate Eye Centers", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
