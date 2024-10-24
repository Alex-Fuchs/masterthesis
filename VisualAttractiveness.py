import argparse
import sys

from pathlib import Path
from typing import List

import clip
from deepface import DeepFace
from ultralytics import YOLO

import numpy as np
import torch
import torch.nn as nn

import cv2
from PIL import Image


class VisualAttractiveness(nn.Module):

    def __init__(self, device="mps"):
        super().__init__()
        self.device = device

        self.predictor, self.predictor_preprocess = clip.load("ViT-B/32", device=device)
        self.human_detector = YOLO("yolov8n.pt")

    @torch.no_grad()
    def predict(self, image: Image.Image, labels: List[str]):
        labels = clip.tokenize(labels).to(self.device)
        text_features = self.predictor.encode_text(labels)
        text_features /= text_features.norm(dim=1, keepdim=True)

        image = self.predictor_preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.predictor.encode_image(image)
        image_features /= image_features.norm(dim=1, keepdim=True)

        logit_scale = self.predictor.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return probs

    def predict_facial_beauty(self, img: np.array, captions):
        faces = DeepFace.extract_faces(img, detector_backend='retinaface', expand_percentage=10, normalize_face=False)

        if len(faces) == 1:
            face = Image.fromarray(faces[0]['face'].astype('uint8'), 'RGB')

            if len(captions) == 2:
                gender = self.predict(face, ["man", "woman"]).argmax()
                score = self.predict(face, captions[gender])[0][0] * 10
            else:
                score = self.predict(face, captions[0])[0][0] * 10

            return score, face, False
        else:
            return None, None, True

    def predict_physical_beauty(self, img: np.array, captions):
        detections = self.human_detector(img)[0].boxes.data.cpu().tolist()
        human_detections = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf, cid in detections if cid == 0]

        if len(human_detections) == 1:
            x1, y1, x2, y2 = human_detections[0]

            human = img[round(y1):round(y2), round(x1):round(x2)]
            human = cv2.cvtColor(human, cv2.COLOR_BGR2RGB)
            human = Image.fromarray(human.astype('uint8'), 'RGB')

            if len(captions) == 2:
                gender = self.predict(human, ["man", "woman"]).argmax()
                score = self.predict(human, captions[gender])[0][0] * 10
            else:
                score = self.predict(human, captions[0])[0][0] * 10

            return score, human, False
        else:
            return None, None, True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AttractiveMeter")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to image"
    )

    args = parser.parse_args()
    image_path = args.image_path

    if not Path(image_path).exists():
        sys.exit(f"File {image_path} does not exist")

    predictor = VisualAttractiveness()

    image = cv2.imread(image_path)

    facial_beauty_score, face, _ = predictor.predict_facial_beauty(image, [["attractive", "unattractive"]])
    human_beauty_score, human, _ = predictor.predict_physical_beauty(image, [["attractive", "unattractive"]])

    print(f'FB: {facial_beauty_score}, HB: {human_beauty_score}')
    face.show()
    human.show()
