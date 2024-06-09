import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from train import UNet
import numpy as np 

# Load the trained model
model_path = '/teamspace/studios/this_studio/Aerial-Segmentation/model.pth'
model = UNet(n_channels=3, n_classes=6)
model.load_state_dict(torch.load(model_path))
model.eval()

# Create a Streamlit app
st.title('Aerial Image Segmentation')

# Add a file uploader to the app
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display the original image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    data_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()]
    )
    image = data_transform(image)
    image = image.unsqueeze(0)  # add a batch dimension

    # Pass the image through the model
    with torch.no_grad():
        output = model(image)

    # Postprocess the output
    # Define the color map
    color_map = {
        0: np.array([155, 155, 155]),  # Unlabeled
        1: np.array([60, 16, 152]),  # Building
        2: np.array([132, 41, 246]),  # Land
        3: np.array([110, 193, 228]),  # Road
        4: np.array([254, 221, 58]),  # Vegetation
        5: np.array([226, 169, 41])  # Water
    }
    class_labels = {
    0: 'Unlabeled',
    1: 'Building',
    2: 'Land',
    3: 'Road',
    4: 'Vegetation',
    5: 'Water'
    }

    # Display the class labels and their colors in a sidebar
    for k, v in class_labels.items():
        st.sidebar.markdown(f'<div style="color:rgb{tuple(color_map[k])};">{v}</div>', unsafe_allow_html=True)

    # Pass the image through the model
    with torch.no_grad():
        output = model(image)

    # Postprocess the output
    output = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    # Squeeze the batch dimension
    output = np.squeeze(output)

    # Now you can create the RGB image
    output_rgb = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for k, v in color_map.items():
        output_rgb[output == k] = v

    # Display the segmented image
    st.image(output_rgb, caption='Segmented Image.', use_column_width=True)


