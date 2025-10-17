import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# -------------------------------
# Load Pretrained ImageNet Model
# -------------------------------
model = ResNet50(weights="imagenet")

# -------------------------------
# Create Tkinter Window
# -------------------------------
root = tk.Tk()
root.title("Cat & Dog Classifier (Deep Learning - ResNet50)")
root.geometry("900x700")
root.configure(bg="#f0f0f0")

# -------------------------------
# UI Components
# -------------------------------
title_label = tk.Label(root, text="🐶🐱 ImageNet Cat-Dog Classifier", font=("Arial", 20, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

frame = tk.Frame(root, bg="#ffffff", relief="solid", bd=1)
frame.pack(padx=10, pady=10, fill="both", expand=True)

img_label = tk.Label(frame, bg="#ffffff")
img_label.pack(pady=10)

result_label = tk.Label(frame, text="", font=("Arial", 14), bg="#ffffff")
result_label.pack(pady=10)

misclassified_dogs = []

# -------------------------------
# Functions
# -------------------------------
def classify_image(img_path):
    """Run ImageNet classification on given image."""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=1)[0]  # CHANGED: top=3 to top=1
    return decoded


def upload_images():
    """Upload multiple images and classify them."""
    global misclassified_dogs
    misclassified_dogs = []

    file_paths = filedialog.askopenfilenames(
        title="Select Images",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_paths:
        return

    for path in file_paths:
        # Display image
        img = Image.open(path)
        img = img.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

        # Predict
        preds = classify_image(path)
        # CHANGED: Only show top-1 prediction
        top_pred = preds[0]
        result_text = f"Top Prediction:\n{top_pred[1]} ({top_pred[2]*100:.2f}%)"

        # Check if dog misclassified
        filename = path.split("/")[-1]
        if "dog" in filename.lower():
            # CHANGED: Check only the top prediction
            top_label = top_pred[1].lower()
            if "dog" not in top_label:
                misclassified_dogs.append((filename, top_pred[1], top_pred[2]))

        result_label.configure(text=result_text)
        root.update()
        root.after(1500)  # wait 1.5s before showing next image

    # Summary popup
    if misclassified_dogs:
        msg = "\n".join([f"{n}: predicted as {p} ({c*100:.1f}%)" for n, p, c in misclassified_dogs])
        messagebox.showwarning("Misclassified Dogs", f"🐾 Misclassified Dog Images:\n\n{msg}")
    else:
        messagebox.showinfo("Results", "✅ All dog images classified correctly!")


# -------------------------------
# Buttons
# -------------------------------
upload_btn = tk.Button(root, text="📂 Upload Images", command=upload_images, font=("Arial", 14), bg="#4CAF50", fg="white")
upload_btn.pack(pady=10)

quit_btn = tk.Button(root, text="❌ Quit", command=root.destroy, font=("Arial", 12), bg="#e74c3c", fg="white")
quit_btn.pack(pady=10)

# -------------------------------
# Run GUI
# -------------------------------
root.mainloop()