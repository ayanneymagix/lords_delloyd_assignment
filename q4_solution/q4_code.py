import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Capture video from webcam (0 = default webcam, change index if needed)
cap = cv2.VideoCapture(0)

# Video writer setup (will initialize when saving starts)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
saving = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Blur detected faces
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        roi_blurred = cv2.GaussianBlur(roi, (99, 99), 30)   # strong blur
        frame[y:y+h, x:x+w] = roi_blurred

    # If saving, write the frame
    if saving and out is not None:
        out.write(frame)

    # Show the output
    cv2.imshow("Face Blurring - Press 'q' to quit, 's' to save", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):   # quit
        break
    elif key == ord('s'):  # toggle saving
        if not saving:
            # Start saving
            out = cv2.VideoWriter("output.avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            saving = True
            print("Started saving video...")
        else:
            # Stop saving
            saving = False
            out.release()
            out = None
            print("Stopped saving video.")

# Release resources
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
