import numpy as np
import cv2
from PIL import Image
from retinaface import RetinaFace


def extract_face(image: Image.Image, target_size=(224, 224)):

    # Convert PIL → numpy
    img = np.array(image)

    # Detect faces using RetinaFace
    detections = RetinaFace.detect_faces(img)

    # If no face detected
    if detections is None or len(detections) == 0:
        return None

    # Get the largest detected face
    largest_face = None
    max_area = 0

    for key in detections:
        x1, y1, x2, y2 = detections[key]["facial_area"]
        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            largest_face = (x1, y1, x2, y2)

    x1, y1, x2, y2 = largest_face

    # Add padding (20%)
    w = x2 - x1
    h = y2 - y1
    pad_w = int(0.2 * w)
    pad_h = int(0.2 * h)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img.shape[1], x2 + pad_w)
    y2 = min(img.shape[0], y2 + pad_h)

    # Crop face
    face = img[y1:y2, x1:x2]

    # Resize to model input
    face = cv2.resize(face, target_size)

    # Convert back to PIL
    face = Image.fromarray(face)

    return face