import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.preprocessing import image

# 1️⃣ Load the trained model
model = tf.keras.models.load_model(r"F:\Work\SEM3\DLI\Potato-Disease\potatoes.h5")

# ✅ Build the model by calling it once (this fixes the “no defined output” error)
_ = model(tf.zeros((1, 256, 256, 3)))  # just a dummy input to initialize the model

# 2️⃣ Load class names (must match training order)
class_names = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

# 3️⃣ Dataset path
base_dir = r"F:\Work\SEM3\DLI\Potato-Disease\training\PlantVillage"

# Pick random image
random_class = random.choice(os.listdir(base_dir))
class_path = os.path.join(base_dir, random_class)
random_img = random.choice(os.listdir(class_path))
img_path = os.path.join(class_path, random_img)

print(f"🖼️ Selected random image: {img_path}")
print(f"✅ True Label: {random_class}")

# 4️⃣ Preprocess the image
IMG_SIZE = (256, 256)
img = image.load_img(img_path, target_size=IMG_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# 5️⃣ Get model prediction
preds = model.predict(x)
pred_class = np.argmax(preds[0])
pred_label = class_names[pred_class]
confidence = round(100 * np.max(preds[0]), 2)
print(f"🔮 Predicted Label: {pred_label} ({confidence}% confidence)")

# 6️⃣ Grad-CAM
# 🔍 Get the last Conv2D layer name (run model.summary() to confirm)
last_conv_layer_name = 'conv2d_11'  # update if needed

grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

# Compute Gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(x)
    loss = predictions[:, pred_class]

grads = tape.gradient(loss, conv_outputs)[0]
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# 7️⃣ Show Grad-CAM heatmap
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.title(f"Grad-CAM\nTrue: {random_class}\nPredicted: {pred_label} ({confidence}%)")
plt.axis("off")
plt.show()
