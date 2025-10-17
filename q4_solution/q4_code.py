import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Face Detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
saving = False

# Initialize Mediapipe Face Detector
with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB (Mediapipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        # If faces detected
        if results.detections:
            for det in results.detections:
                bboxC = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)

                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                bw = min(w - x, bw)
                bh = min(h - y, bh)

                # Blur only the detected face region
                face_roi = frame[y:y+bh, x:x+bw]
                if face_roi.size > 0:
                    face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    frame[y:y+bh, x:x+bw] = face_roi

        # Save if toggled
        if saving and out is not None:
            out.write(frame)

        # Display
        cv2.imshow("Face-Only Deep Learning Blur (Press 's' to save, 'q' to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if not saving:
                h, w = frame.shape[:2]

                # ✅ Save video in same folder as this script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                save_path = os.path.join(script_dir, "output_faceonly.avi")

                out = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))
                saving = True
                print(f"🎥 Started saving video at: {save_path}")
            else:
                saving = False
                out.release()
                out = None
                print("🛑 Stopped saving video.")

# Cleanup
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
