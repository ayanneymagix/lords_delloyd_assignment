import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Initialize models
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection
    face_results = face_detection.process(rgb)
    if face_results.detections:
        for detection in face_results.detections:
            # Draw bounding box manually
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 255), 2)

    # Landmark detection
    mesh_results = face_mesh.process(rgb)
    if mesh_results.multi_face_landmarks:
        for landmarks in mesh_results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Nose tip
            nose = landmarks.landmark[1]
            cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 0), -1)

            # Left & right eye centers
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            cv2.circle(frame, (int(left_eye.x * w), int(left_eye.y * h)), 6, (255, 0, 0), -1)
            cv2.circle(frame, (int(right_eye.x * w), int(right_eye.y * h)), 6, (255, 0, 0), -1)

    cv2.imshow("Face + Features", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
