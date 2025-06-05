import streamlit as st # for front end
import cv2 # backend
import numpy as np # for images processing becuase every image is represented as matrix===> mathematical operation

# Title and Description
st.title("Image Editor using OpenCV and Streamlit")
st.write("Upload an image and apply various editing operations such as crop, resize, blur, negative, sketch, and more.")

# File Upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Sidebar header is always visible
st.sidebar.header("Editing Options")

if uploaded_file is not None:
    try:
        # Convert the uploaded file to a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Resize Option
        if st.sidebar.checkbox("Resize Image"):
            width = st.sidebar.number_input("Width", min_value=10, max_value=2000, value=image.shape[1])
            height = st.sidebar.number_input("Height", min_value=10, max_value=2000, value=image.shape[0])
            image = cv2.resize(image, (int(width), int(height)))
            st.image(image, caption='Resized Image', use_column_width=True)
        
        # Crop Option
        if st.sidebar.checkbox("Crop Image"):
            x_start = st.sidebar.number_input("X Start", min_value=0, max_value=image.shape[1] - 1, value=0)
            y_start = st.sidebar.number_input("Y Start", min_value=0, max_value=image.shape[0] - 1, value=0)
            x_end = st.sidebar.number_input("X End", min_value=x_start + 1, max_value=image.shape[1], value=image.shape[1])
            y_end = st.sidebar.number_input("Y End", min_value=y_start + 1, max_value=image.shape[0], value=image.shape[0])
            image = image[int(y_start):int(y_end), int(x_start):int(x_end)]
            st.image(image, caption='Cropped Image', use_column_width=True)

        # Blur Option
        if st.sidebar.checkbox("Apply Blur"):
            blur_intensity = st.sidebar.slider("Blur Intensity", min_value=1, max_value=50, value=5)
            image = cv2.GaussianBlur(image, (blur_intensity * 2 + 1, blur_intensity * 2 + 1), 0)
            st.image(image, caption='Blurred Image', use_column_width=True)

        # Negative Effect
        if st.sidebar.checkbox("Apply Negative Effect"):
            image = cv2.bitwise_not(image)
            st.image(image, caption='Negative Image', use_column_width=True)

        # Sketch Effect
        if st.sidebar.checkbox("Apply Sketch Effect"):
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            inverted_image = cv2.bitwise_not(gray_image)
            blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)
            sketch_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)
            st.image(sketch_image, caption='Sketch Image', use_column_width=True, clamp=True)

        # Brightness and Contrast
        if st.sidebar.checkbox("Adjust Brightness and Contrast"):
            brightness = st.sidebar.slider("Brightness", min_value=-100, max_value=100, value=0)
            contrast = st.sidebar.slider("Contrast", min_value=-100, max_value=100, value=0)
            image = cv2.convertScaleAbs(image, alpha=1 + contrast / 100, beta=brightness)
            st.image(image, caption='Brightness/Contrast Adjusted Image', use_column_width=True)
        
        # Grayscale Effect
        if st.sidebar.checkbox("Convert to Grayscale"):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            st.image(image, caption='Grayscale Image', use_column_width=True, clamp=True)
        
        # Rotate Image
        if st.sidebar.checkbox("Rotate Image"):
            rotation_angle = st.sidebar.slider("Rotation Angle", min_value=-180, max_value=180, value=0)
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
            st.image(image, caption='Rotated Image', use_column_width=True)
        
        # Save Edited Image
        if st.sidebar.button("Save Image"):
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('edited_image.jpg', result)
            st.success("Image saved as 'edited_image.jpg'")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
else:
    st.warning("Please upload an image file to start editing.")
    st.sidebar.write("Upload an image to see available options.")
