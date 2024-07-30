import streamlit as st
import numpy as np
from PIL import Image
from ISR.models import RDN
import io

# Initialize the RDN model
rdn = RDN(weights='psnr-small')

# Function to enhance image
def enhance_image(lr_img):
    try:
        sr_img = rdn.predict(lr_img)
    except Exception as e1:
        try:
            sr_img = rdn.predict(lr_img, by_patch_of_size=50)
        except Exception as e2:
            return f"Error: Enhancement failed with both methods. First error: {str(e1)}. Second error: {str(e2)}."
    return sr_img

# Streamlit app
st.title('Image Enhancement with RDN')

# File uploader
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)

if uploaded_files:
    # Process and display each uploaded image
    for uploaded_file in uploaded_files:
        # Read the image
        img = Image.open(uploaded_file)
        lr_img = np.array(img)
        
        # Show the original image
        st.image(img, caption='Original Image', use_column_width=True)
        st.write("Image read successfully")
        
        # Enhance the image
        st.write("Enhancing image...")
        sr_img = enhance_image(lr_img)
        
        if isinstance(sr_img, str):  # Error message
            st.error(sr_img)
        else:
            # Convert numpy array back to PIL Image
            sr_img = Image.fromarray(sr_img)
            
            # Save the enhanced image to a BytesIO object
            enhanced_img_io = io.BytesIO()
            sr_img.save(enhanced_img_io, format='JPEG')
            enhanced_img_io.seek(0)
            
            # Display the enhanced image
            st.image(sr_img, caption='Enhanced Image', use_column_width=True)
            st.write("Enhancement complete")
