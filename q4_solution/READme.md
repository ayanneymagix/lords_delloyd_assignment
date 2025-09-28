# Face Blurring in Video Feeds

## Project Overview

This project captures a **live video feed** from a webcam or CCTV and automatically **detects and blurs all faces** in real-time. Additionally, it allows the user to **save a video clip** when required.

This is useful for **privacy protection**, surveillance masking, or anonymizing faces in video recordings.

**Type of Computer Vision Problem:**

* **Face Detection:** Detects faces using Haar cascades (object detection).
* **Image Processing:** Applies Gaussian blur to detected face regions.

---

## File Structure

```
├── face_blurring_video.py   # Main Python script for live face blurring
├── requirements.txt         # Python dependencies
└── README.md                # Project overview and instructions
```

---

## Dependencies

Install required packages:

```bash
pip install opencv-python
```

Optional:

* `numpy` (usually installed with OpenCV)

---

## How to Run

1. Open `face_blurring_video.py` in your IDE or terminal.
2. Ensure your webcam is connected (default device `0`) or update the index for another camera.
3. Run the script:

```bash
python face_blurring_video.py
```

4. Functionality:

   * **Live Face Blurring:** Detected faces are blurred with a strong Gaussian blur.
   * **Toggle Video Saving:**

     * Press **`s`** to start saving the processed video (`output.avi`).
     * Press **`s`** again to stop saving.
   * **Quit Application:** Press **`q`** to exit.

5. The output video will match the live feed dimensions and display all detected faces blurred.

---

## Code Highlights

* **Face Detection:**

```python
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
```

Detects faces in the grayscale frame using Haar cascade classifier.

* **Blurring Faces:**

```python
roi_blurred = cv2.GaussianBlur(roi, (99, 99), 30)
frame[y:y+h, x:x+w] = roi_blurred
```

Applies Gaussian blur to the region of interest containing the face.

* **Video Saving:**

```python
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
```

Initializes a video writer to save processed frames.

---

## Notes & Tips

* Adjust `scaleFactor`, `minNeighbors`, and `minSize` for better face detection in different lighting or camera setups.
* Increase or decrease the **Gaussian blur kernel** `(99, 99)` for stronger or lighter face anonymization.
* Ensure sufficient storage if saving long video clips.
* Can be extended to multiple cameras or CCTV streams by changing the `VideoCapture` index or URL.

---
