from __future__ import annotations

import cv2
import mediapipe as mp 
import numpy as np
import torch

from torchvision import transforms


class FaceHelper:

    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.5,
        image_size: int = 224,
        margin: float = 0.20,
        device: str | torch.device = "cpu",
    ):

        self.image_size = image_size
        self.margin = margin
        self.device = torch.device(device)

        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def detect(
        self,
        frame: np.ndarray,
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = self.face_detector.process(rgb)

        if results.detections is None:
            return []

        return results.detections

    def detect_largest(
        self,
        frame: np.ndarray,
    ):

        detections = self.detect(frame)

        if len(detections) == 0:
            return None

        return max(
            detections,
            key=lambda d:
            d.location_data.relative_bounding_box.width *
            d.location_data.relative_bounding_box.height
        )
    @staticmethod
    def _bbox_from_detection(
        detection,
        width: int,
        height: int,
        margin: float,
    ):

        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)

        w = int(bbox.width * width)
        h = int(bbox.height * height)

        pad_x = int(w * margin)
        pad_y = int(h * margin)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        return x1, y1, x2, y2

    def get_bbox(
        self,
        frame: np.ndarray,
    ):

        detection = self.detect_largest(frame)

        if detection is None:
            return None

        h, w = frame.shape[:2]

        return self._bbox_from_detection(
            detection,
            w,
            h,
            self.margin,
        )

    def crop_face(
        self,
        frame: np.ndarray,
    ):

        bbox = self.get_bbox(frame)

        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))

        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            return None

        return face

    def resize_face(
        self,
        face: np.ndarray,
    ):

        return cv2.resize(
            face,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_AREA,
        )
    def to_tensor(
        self,
        face: np.ndarray,
    ):

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB,
        )

        tensor = self.transform(face)

        return tensor.to(
            self.device,
            non_blocking=True,
        )

    def process_frame(
        self,
        frame: np.ndarray,
    ):

        face = self.crop_face(frame)

        if face is None:
            raise RuntimeError("No face detected.")

        face = self.resize_face(face)

        return self.to_tensor(face)

    def process_frames(
        self,
        frames,
    ):

        output = []

        previous_face = None

        for frame in frames:

            try:

                face = self.crop_face(frame)

                if face is None:

                    if previous_face is None:
                        continue

                    face = previous_face.copy()

                else:

                    previous_face = face.copy()

                face = self.resize_face(face)

                output.append(
                    self.to_tensor(face)
                )

            except Exception:

                if previous_face is not None:

                    face = self.resize_face(
                        previous_face
                    )

                    output.append(
                        self.to_tensor(face)
                    )

        return output

    def close(self):

        self.face_detector.close()