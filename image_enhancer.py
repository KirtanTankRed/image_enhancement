import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from tensorflow.contrib import slim

# Define the lrelu activation function
def lrelu(x):
    return tf.maximum(x*0.2, x)

# Define the identity initializer
def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0] // 2, shape[1] // 2
        for i in range(np.minimum(shape[2], shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

# Define the normalization function
def nm(x):
    w0 = tf.Variable(1.0, name='w0')
    w1 = tf.Variable(0.0, name='w1')
    return w0 * x + w1 * slim.batch_norm(x)

# Define the network architecture
def build(input):
    net = slim.conv2d(input, 24, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv1')
    net = slim.conv2d(net, 24, [3, 3], rate=2, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv2')
    net = slim.conv2d(net, 24, [3, 3], rate=4, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv3')
    net = slim.conv2d(net, 24, [3, 3], rate=8, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv4')
    net = slim.conv2d(net, 24, [3, 3], rate=16, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv5')
    net = slim.conv2d(net, 24, [3, 3], rate=32, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv6')
    net = slim.conv2d(net, 24, [3, 3], rate=64, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv7')
    net = slim.conv2d(net, 24, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='g_conv9')
    net = slim.conv2d(net, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv_last')
    return net

# Load model
sess = tf.Session()
input_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3])
network = build(input_ph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state("L0_smoothing")  # Adjust path as needed
if ckpt:
    saver.restore(sess, ckpt.model_checkpoint_path)

# Streamlit interface
st.title("Image Processing with Pretrained Model")
st.write("Upload an image and apply the task.")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))
    image = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

    # Process image
    output_image = sess.run(network, feed_dict={input_ph: image})
    output_image = np.clip(output_image, 0.0, 1.0) * 255.0
    output_image = output_image[0].astype(np.uint8)

    # Display images
    st.image([image[0].astype(np.uint8), output_image], caption=['Original', 'Processed'], width=300)
