import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO

# Define lrelu activation function
def lrelu(x):
    return tf.maximum(x * 0.2, x)

# Define custom initializer
def identity_initializer(shape, dtype=tf.float32):
    array = np.zeros(shape, dtype=float)
    cx, cy = shape[0] // 2, shape[1] // 2
    for i in range(np.minimum(shape[2], shape[3])):
        array[cx, cy, i, i] = 1
    return tf.constant(array, dtype=dtype)

# Define normalization method
def nm(x):
    w0 = tf.Variable(1.0, name='w0')
    w1 = tf.Variable(0.0, name='w1')
    return w0 * x + w1 * tf.keras.layers.BatchNormalization()(x)

# Build the model
def build(input_tensor):
    initializer = tf.keras.initializers.Constant(identity_initializer([3, 3, 3, 24]))
    x = tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation=lrelu, kernel_initializer=initializer)(input_tensor)
    for rate in [2, 4, 8, 16, 32, 64, 1]:
        x = tf.keras.layers.Conv2D(24, (3, 3), dilation_rate=rate, padding='same', activation=lrelu, kernel_initializer=initializer)(x)
    x = tf.keras.layers.Conv2D(3, (1, 1), padding='same')(x)
    return x

# Streamlit app
st.title("Image Processing with TensorFlow")

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

task = "L0_smoothing"
is_training = False

# Placeholder for input and output
input_tensor = tf.keras.Input(shape=(None, None, 3))
output_tensor = build(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# Load pre-trained model
ckpt = tf.train.Checkpoint(model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, task, max_to_keep=1000)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    st.write('Loaded model: ' + ckpt_manager.latest_checkpoint)

if uploaded_files:
    st.write("Processing images...")

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        input_image = np.expand_dims(image, axis=0) / 255.0

        st.image(image, caption='Original Image', use_column_width=True)

        output_image = model.predict(input_image)
        output_image = np.minimum(np.maximum(output_image, 0.0), 1.0) * 255.0
        output_image = np.uint8(output_image[0])

        st.image(output_image, caption='Processed Image', use_column_width=True)

        # Convert output image to bytes
        is_success, buffer = cv2.imencode(".jpg", output_image)
        io_buf = BytesIO(buffer)

        st.download_button(label="Download Processed Image", data=io_buf, file_name="processed_image.jpg", mime="image/jpeg")

st.write("Upload images to process them.")

