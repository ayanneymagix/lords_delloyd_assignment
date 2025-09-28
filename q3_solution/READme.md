# Face Detection and Feature Localization

## Project Overview

This project detects faces in real-time video streams or images and localizes key facial features, specifically:

* **Tip of the nose**
* **Centers of the left and right eyes**

The system annotates these features on the video feed. It uses **MediaPipe** for face detection and face mesh landmark localization.

**Type of Computer Vision Problem:**
This is a **Face Landmark Detection** problem, which falls under **object detection + keypoint localization**.

**Justification:**

* The model first detects faces (object detection).
* Then it identifies precise facial landmarks (keypoint localization).
* Each landmark (nose tip, eye centers) is represented by coordinates in the image, which is typical of a **keypoint regression task** in computer vision.

---

## File Structure

```
├── face_detection_feature_localization.py   # Main Python script for real-time face detection
└── README.md                                # Project overview and instructions
```

---

## Dependencies

Install required packages:

```bash
pip install opencv-python mediapipe
```

Optional:

* `numpy` (if additional numerical operations are needed)

---

## How to Run

1. Open `face_detection_feature_localization.py` in your IDE or terminal.
2. Ensure your webcam is connected (the script uses `cv2.VideoCapture(0)` for live feed).
3. Run the script:

```bash
python face_detection_feature_localization.py
```

4. A window will appear showing:

   * **White bounding boxes** around detected faces
   * **Green circle:** tip of the nose
   * **Blue circles:** centers of the left and right eyes

5. Press **ESC** to exit the application.

---

## Code Highlights

* **Face Detection:**

```python
face_results = face_detection.process(rgb)
```

Detects faces and outputs bounding boxes.

* **Facial Landmark Detection:**

```python
mesh_results = face_mesh.process(rgb)
```

Finds key landmarks like the nose tip and eye centers.

* **Annotation:**

```python
cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 0), -1)
```

Draws a circle on each landmark point for visualization.

---

## Notes & Tips

* Adjust `min_detection_confidence` in `FaceDetection` for more or less sensitive face detection.
* The eye and nose landmarks use **MediaPipe’s 468-point mesh**, so specific indices correspond to different facial features.

  * Nose tip: landmark `[1]`
  * Left eye center: landmark `[33]`
  * Right eye center: landmark `[263]`
* For higher accuracy or multiple faces, you can refine the detection parameters or loop through multiple detected faces.

---

