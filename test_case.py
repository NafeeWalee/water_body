import cv2
import numpy as np
import os
from keras.models import load_model

DIR = rf"{os.path.dirname(__file__)}"

noMask_Image= rf"{DIR}\datasets\noMask\water_body_3.jpg"

resized_image = cv2.resize(noMask_Image, (150, 150), interpolation=cv2.INTER_AREA)
normalized_image = resized_image / 255.0  # Normalize pixel values to range [0, 1]

# Load the pre-trained model
model = load_model('water_detection_model.h5')

# Load the new image
image = cv2.imread(normalized_image)

# Predict the mask using the model
predicted_mask = model.predict(np.expand_dims(image, axis=0))[0]

# Threshold the predicted mask to get binary mask
threshold = 0.5  # You may need to adjust this threshold based on your model's output
binary_mask = (predicted_mask > threshold).astype(np.uint8) * 255

# Apply the binary mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=binary_mask)

# Show the masked image
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
