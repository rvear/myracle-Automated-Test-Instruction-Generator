import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import openai
import os
from dotenv import load_dotenv

# Loading the environment variables from the .env file
load_dotenv()

# Geting OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Loading a pre-trained YOLOv5 model 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  

# Streamlit app interface
st.title("Automated Test Instruction Generator")

st.write("""
### Upload a screenshot of a digital product and get test instructions for its features.
""")

# (screenshot)
uploaded_image = st.file_uploader("Choose a screenshot", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Converting the uploaded image to an OpenCV image
    image = np.array(Image.open(uploaded_image))
    
    #  uploaded image
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)
    
    st.write("Processing image...")

    
    results = model(image)

    # List to store detected objects
    detected_objects = []

    # Iterating over detected objects and extract labels
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]  
        detected_objects.append(label)

    
    st.write("Detected objects:", detected_objects)

    # Generating a prompt for GPT-3.5-turbo based on detected objects
    if len(detected_objects) > 0:
        prompt = f"You are testing a screen with the following detected objects: {', '.join(detected_objects)}.\nPlease generate detailed test instructions for each component."

        # Calling GPT-3.5-turbo to generate test instructions(as it will be cheaper and perform good enough)
        if st.button("Generate Test Instructions"):
            with st.spinner("Generating instructions..."):
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500
                    )
                    instructions = response['choices'][0]['message']['content'].strip()

                    # Displaying the generated test instructions
                    st.write("### Generated Test Instructions")
                    st.write(instructions)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.write("No objects detected in the image. Please upload a valid screenshot.")
