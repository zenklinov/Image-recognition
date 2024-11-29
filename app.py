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
st.write("What is ResNet50?")
st.markdown("""
ResNet-50 is a deep convolutional neural network that is part of the ResNet (Residual Network) family, which was introduced by researchers at Microsoft in the 2015 paper "Deep Residual Learning for Image Recognition". It is designed to address the problem of vanishing gradients and degradation of accuracy in very deep networks. ResNet-50 is a 50-layer version of this architecture, widely used for image classification tasks.

### Labels:
The model can classify a wide variety of categories, including:

- **Animals**: Both terrestrial and aquatic species, such as birds (e.g., "ostrich," "flamingo"), reptiles (e.g., "Komodo dragon," "Nile crocodile"), amphibians, fish (e.g., "great white shark," "clownfish"), mammals (e.g., "lion," "chimpanzee"), and insects (e.g., "dragonfly," "ant").
- **Dog Breeds**: A comprehensive list of dog breeds, from small toy breeds (e.g., "Chihuahua") to working and hound breeds (e.g., "German Shepherd Dog," "Dobermann").
- **Objects**: Everyday items like "backpack," "toaster," "vacuum cleaner," and "computer keyboard."
- **Vehicles**: Land, air, and water vehicles such as "aircraft carrier," "taxicab," and "canoe."
- **Buildings and Structures**: Including "barn," "castle," "mosque," and "lighthouse."
- **Clothing and Accessories**: Items like "kimono," "jeans," "bow tie," and "sunglasses."
- **Musical Instruments**: Examples include "guitar," "accordion," and "violin."
- **Food and Beverages**: A range of dishes ("pizza," "carbonara") and produce ("broccoli," "pineapple") to drinks ("espresso," "red wine").
- **Natural Features**: Landforms and landscapes such as "volcano," "seashore," and "geyser."
- **Miscellaneous Items**: Including "traffic light," "menu," "soap dispenser," and "bagel."
""")

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
