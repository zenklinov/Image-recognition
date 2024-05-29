import torch
from torchvision import models, transforms
from PIL import Image, ImageTk
import urllib.request
import json
import tkinter as tk
from tkinter import filedialog, Label, Button, Entry

# Load the pre-trained ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Function to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load labels from ImageNet
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(LABELS_URL) as url:
    labels = json.loads(url.read().decode())

# Function to open file dialog and select an image
def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_path_entry.delete(0, tk.END)
        img_path_entry.insert(0, file_path)

# Function to process the image and perform prediction
def process_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        out = model(batch_t)
    
    _, indices = torch.topk(out, k=5)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    result_text = "Classified as:\n"
    for idx in indices[0]:
        result_text += f"{labels[idx]} ({percentage[idx].item():.2f}%)\n"
    result_label.config(text=result_text)
    
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep a reference to avoid garbage collection

# Function to download and process an image from a URL
def browse_image_from_url():
    url = url_entry.get()
    if url:
        try:
            with urllib.request.urlopen(url) as response:
                img = Image.open(response).convert('RGB')
                img_path = 'downloaded_image.jpg'
                img.save(img_path)
                process_image(img_path)
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")

# Open LinkedIn profile
def open_linkedin():
    import webbrowser
    webbrowser.open_new("https://www.linkedin.com/in/zenklinov/")

# Open GitHub profile
def open_github():
    import webbrowser
    webbrowser.open_new("https://github.com/zenklinov/")

# Open Instagram profile
def open_instagram():
    import webbrowser
    webbrowser.open_new("https://instagram.com/zenklinov")

# Create the main window
root = tk.Tk()
root.title("Image Classifier Resnet 50, Update 29 May 2024")

# Creator information
creator_frame = tk.Frame(root)
creator_frame.pack(anchor=tk.E, padx=10, pady=5, fill='x')

creator_label = Label(creator_frame, text="Creator: Amanatullah Pandu Zenklinov")
creator_label.pack(side=tk.LEFT)

linkedin_button = Button(creator_frame, text="LinkedIn", command=open_linkedin)
linkedin_button.pack(side=tk.LEFT, padx=5)

github_button = Button(creator_frame, text="GitHub", command=open_github)
github_button.pack(side=tk.LEFT, padx=5)

instagram_button = Button(creator_frame, text="Instagram", command=open_instagram)
instagram_button.pack(side=tk.LEFT, padx=5)

# Create and pack the widgets for local file selection
local_file_frame = tk.Frame(root)
local_file_frame.pack(anchor=tk.W, padx=10, pady=5, fill='x')

local_file_label = Label(local_file_frame, text="Choose from local file:")
local_file_label.pack(side=tk.LEFT)

img_path_entry = Entry(local_file_frame, width=50)
img_path_entry.pack(side=tk.LEFT, padx=5)

browse_button = Button(local_file_frame, text="Browse", command=browse_image)
browse_button.pack(side=tk.LEFT, padx=5)

local_check_button = Button(local_file_frame, text="Check", command=lambda: process_image(img_path_entry.get()))
local_check_button.pack(side=tk.LEFT)

# Create and pack the widgets for URL input
url_frame = tk.Frame(root)
url_frame.pack(anchor=tk.W, padx=10, pady=5, fill='x')

url_label = Label(url_frame, text="Choose from URL:")
url_label.pack(side=tk.LEFT)

url_entry = Entry(url_frame, width=50)
url_entry.pack(side=tk.LEFT, padx=5)

url_check_button = Button(url_frame, text="Check", command=browse_image_from_url)
url_check_button.pack(side=tk.LEFT)

# Create and pack the label to display the supported file formats
supported_formats_label = Label(root, text="Supported file formats: JPEG, PNG, BMP, GIF, TIFF")

supported_formats_label.pack(padx=10, pady=5)

# Create and pack the label to display the selected image
image_frame = tk.Frame(root)
image_frame.pack(anchor=tk.CENTER, padx=10, pady=10)

image_result_label = Label(image_frame, text="Image result:")
image_result_label.pack()

image_label = Label(image_frame)
image_label.pack()

# Create and pack the label to display the prediction results
result_label = Label(root, text="", anchor=tk.CENTER)
result_label.pack(padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
