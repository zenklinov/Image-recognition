import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request
import json
import streamlit as st
import pyperclip

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

# Function to process the image and perform prediction
def process_image(image):
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        out = model(batch_t)
    
    _, indices = torch.topk(out, k=5)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    results = []
    for idx in indices[0]:
        results.append(f"{labels[idx]} ({percentage[idx].item():.2f}%)")
    return results

# Streamlit app
st.title("Image Classifier ResNet50")
st.write("Creator: Amanatullah Pandu Zenklinov")
st.markdown("""
[LinkedIn](https://www.linkedin.com/in/zenklinov/) | 
[GitHub](https://github.com/zenklinov/) | 
[Instagram](https://instagram.com/zenklinov)
""")

# File upload section
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "bmp", "gif", "tiff"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform classification
    st.write("Classifying...")
    results = process_image(image)

    st.write("### Classification Results")
    for result in results:
        st.write(result)

# URL input section
url = st.text_input("Or enter an image URL:")
if url:
    try:
        with urllib.request.urlopen(url) as response:
            image = Image.open(response).convert('RGB')
            st.image(image, caption="Image from URL", use_column_width=True)

            # Perform classification
            st.write("Classifying...")
            results = process_image(image)

            st.write("### Classification Results")
            for result in results:
                st.write(result)
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")

# Clipboard input section
st.write("### Or use a URL copied to clipboard")
if st.button("Paste from Clipboard"):
    clipboard_content = pyperclip.paste()
    if clipboard_content:
        try:
            with urllib.request.urlopen(clipboard_content) as response:
                image = Image.open(response).convert('RGB')
                st.image(image, caption="Image from Clipboard URL", use_column_width=True)

                # Perform classification
                st.write("Classifying...")
                results = process_image(image)

                st.write("### Classification Results")
                for result in results:
                    st.write(result)
        except Exception as e:
            st.error(f"Error loading image from Clipboard URL: {e}")
    else:
        st.warning("No URL found in clipboard.")
