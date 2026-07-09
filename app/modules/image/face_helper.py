import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T


class FaceHelper:

    def __init__(
        self,
        model_selection=0,
        min_detection_confidence=0.5,
        image_size=224,
        margin=0.20,
        device=None,
    ):

        self.image_size = image_size
        self.margin = margin

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def detect(self, image):

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.face_detector.process(rgb)

        if results.detections is None:
            return []

        return results.detections

    def detect_largest(self, image):

        detections = self.detect(image)

        if len(detections) == 0:
            return None

        return max(
            detections,
            key=lambda d:
            d.location_data.relative_bounding_box.width *
            d.location_data.relative_bounding_box.height,
        )

    def _bbox(self, detection, w, h):

        box = detection.location_data.relative_bounding_box

        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        padx = int(bw * self.margin)
        pady = int(bh * self.margin)

        x1 = max(0, x - padx)
        y1 = max(0, y - pady)
        x2 = min(w, x + bw + padx)
        y2 = min(h, y + bh + pady)

        return x1, y1, x2, y2

    def _detect_face(self, img):

        h, w = img.shape[:2]

        detection = self.detect_largest(img)

        if detection is None:

            size = int(min(h, w) * 0.55)

            cx = w // 2
            cy = h // 2

            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, cx + size // 2)
            y2 = min(h, cy + size // 2)

        else:

            x1, y1, x2, y2 = self._bbox(
                detection,
                w,
                h,
            )

        face = img[y1:y2, x1:x2]

        if face.size == 0:
            face = cv2.resize(img, (224, 224))
        else:
            face = cv2.resize(face, (224, 224))

        return face

    def to_tensor(self, face):

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(rgb)

        return self.transform(pil)

    def process(self, img_path):

        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        face = self._detect_face(img)

        return self.to_tensor(face)

    def process_frame(self, frame):

        face = self._detect_face(frame)

        return self.to_tensor(face)

    def extract_face(self, img_path):

        return self.process(img_path)

    def extract_face_image(self, img_path):

        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        return self._detect_face(img)

    def close(self):

        self.face_detector.close()