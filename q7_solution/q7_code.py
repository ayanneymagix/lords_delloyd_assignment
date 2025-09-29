import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import json
import os

class CatDogClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat vs Dog Classifier")
        self.root.geometry("800x600")
        
        # Initialize model
        self.model = None
        self.class_labels = {}
        self.setup_model()
        
        # GUI elements
        self.setup_gui()
        
    def setup_model(self):
        """Initialize the pre-trained model"""
        try:
            self.model = models.resnet50(pretrained=True)
            self.model.eval()
            self.load_imagenet_labels()
            
            # Preprocessing transforms
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def load_imagenet_labels(self):
        """Load ImageNet class labels"""
        # Basic animal labels for common misclassifications
        self.class_labels = {
            281: 'tabby cat', 282: 'tiger cat', 283: 'persian cat', 284: 'siamese cat', 285: 'egyptian cat',
            151: 'chihuahua', 152: 'japanese spaniel', 153: 'maltese dog', 154: 'pekinese', 155: 'shih-tzu',
            156: 'blenheim spaniel', 157: 'papillon', 158: 'toy terrier', 159: 'rhodesian ridgeback', 
            277: 'red fox', 278: 'kit fox', 279: 'arctic fox', 280: 'grey fox'
        }
        # Add more dog breeds
        dog_breeds = {
            160: 'afghan hound', 161: 'basset', 162: 'beagle', 163: 'bloodhound', 164: 'bluetick',
            165: 'black-and-tan coonhound', 166: 'walker hound', 167: 'english foxhound', 168: 'redbone',
            169: 'borzoi', 170: 'irish wolfhound', 171: 'italian greyhound', 172: 'whippet',
            173: 'ibizan hound', 174: 'norwegian elkhound', 175: 'otterhound', 176: 'saluki',
            177: 'scottish deerhound', 178: 'weimaraner', 179: 'staffordshire bullterrier',
            180: 'american staffordshire terrier', 181: 'bedlington terrier', 182: 'border terrier',
            183: 'kerry blue terrier', 184: 'irish terrier', 185: 'norfolk terrier', 186: 'norwich terrier',
            187: 'yorkshire terrier', 188: 'wire-haired fox terrier', 189: 'lakeland terrier',
            190: 'sealyham terrier', 191: 'airedale', 192: 'cairn', 193: 'australian terrier',
            194: 'dandie dinmont', 195: 'boston bull', 196: 'miniature schnauzer', 197: 'giant schnauzer',
            198: 'standard schnauzer', 199: 'scotch terrier', 200: 'tibetan terrier'
        }
        self.class_labels.update(dog_breeds)
    
    def setup_gui(self):
        """Setup the GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Cat vs Dog Classification", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Image selection
        ttk.Label(main_frame, text="Select Images:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.image_listbox = tk.Listbox(main_frame, width=50, height=8)
        self.image_listbox.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Add Images", 
                  command=self.add_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Analyze Images", 
                  command=self.analyze_images).pack(side=tk.LEFT, padx=5)
        
        # Results area
        ttk.Label(main_frame, text="Results:", font=('Arial', 12, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(20, 5))
        
        # Treeview for results
        columns = ('Image', 'True Label', 'Predicted', 'Confidence', 'Status')
        self.results_tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120)
        
        self.results_tree.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.grid(row=5, column=2, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        # Image preview
        self.image_label = ttk.Label(main_frame, text="Image preview will appear here")
        self.image_label.grid(row=1, column=2, rowspan=4, padx=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind selection event
        self.results_tree.bind('<<TreeviewSelect>>', self.show_selected_image)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.image_paths = []
    
    def add_images(self):
        """Add images to the list"""
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        for file_path in files:
            if file_path not in self.image_paths:
                self.image_paths.append(file_path)
                filename = os.path.basename(file_path)
                self.image_listbox.insert(tk.END, filename)
    
    def clear_images(self):
        """Clear all images from the list"""
        self.image_paths.clear()
        self.image_listbox.delete(0, tk.END)
        self.clear_results()
    
    def clear_results(self):
        """Clear results from treeview"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.image_label.configure(image='', text="Image preview will appear here")
    
    def classify_image(self, image_path):
        """Classify a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_batch)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            
            class_id = top_catid[0].item()
            class_name = self.class_labels.get(class_id, f"class_{class_id}")
            confidence = top_prob[0].item()
            
            return class_name, confidence, image
        
        except Exception as e:
            print(f"Error classifying image: {e}")
            return None, None, None
    
    def is_dog_breed(self, class_name):
        """Check if prediction is a dog breed"""
        dog_indicators = ['dog', 'hound', 'terrier', 'retriever', 'spaniel', 
                         'sheepdog', 'bulldog', 'poodle', 'mastiff', 'setter']
        return any(indicator in class_name.lower() for indicator in dog_indicators)
    
    def analyze_images(self):
        """Analyze all loaded images"""
        if not self.image_paths:
            messagebox.showwarning("Warning", "Please add images first!")
            return
        
        self.clear_results()
        misclassified_count = 0
        
        for image_path in self.image_paths:
            filename = os.path.basename(image_path)
            predicted_class, confidence, image = self.classify_image(image_path)
            
            if predicted_class:
                # Determine true label from filename or assume it's a dog for this assignment
                true_label = "Dog"  # Assuming all images are dogs for this assignment
                
                is_dog_prediction = self.is_dog_breed(predicted_class)
                status = "Correct" if is_dog_prediction else "Misclassified"
                
                if not is_dog_prediction:
                    misclassified_count += 1
                
                # Add to results tree
                self.results_tree.insert('', tk.END, values=(
                    filename,
                    true_label,
                    predicted_class,
                    f"{confidence:.2%}",
                    status
                ), tags=(status,))
        
        # Configure tag colors
        self.results_tree.tag_configure('Misclassified', background='#ffcccc')  # Light red
        self.results_tree.tag_configure('Correct', background='#ccffcc')  # Light green
        
        # Show summary
        messagebox.showinfo("Analysis Complete", 
                          f"Analysis completed!\n\n"
                          f"Total images: {len(self.image_paths)}\n"
                          f"Misclassified dogs: {misclassified_count}\n"
                          f"Accuracy: {(len(self.image_paths) - misclassified_count)/len(self.image_paths):.1%}")
    
    def show_selected_image(self, event):
        """Show the selected image in preview"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.results_tree.item(item, 'values')
        filename = values[0]
        
        # Find the image path
        image_path = None
        for path in self.image_paths:
            if os.path.basename(path) == filename:
                image_path = path
                break
        
        if image_path:
            try:
                # Load and resize image for preview
                image = Image.open(image_path)
                image.thumbnail((300, 300))  # Resize for preview
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
            except Exception as e:
                self.image_label.configure(image='', text=f"Error loading image: {e}")

def main():
    root = tk.Tk()
    app = CatDogClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()