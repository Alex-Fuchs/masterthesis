import argparse
import sys

from pathlib import Path

import clip
from deepface import DeepFace
from ultralytics import YOLO

import torch.nn as nn

import cv2
from PIL import Image


class VisualAttractiveness(nn.Module):

    def __init__(self, captions=None, device="cuda"):
        super().__init__()
        self.device = device

        self.predictor, self.predictor_preprocess = clip.load("ViT-B/32", device=device)
        self.human_detector = YOLO("yolov8n.pt")

        if captions is None:
            self.text_features = self.encode_captions(["attractive", "unattractive"])
        else:
            self.text_features = self.encode_captions(captions)

    def predict_scores(self, image_features, text_features=None):
        if text_features is None:
            text_features = self.text_features

        logit_scale = self.predictor.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        scores = logits_per_image.softmax(dim=-1) * 10

        return scores

    def encode_captions(self, captions):
        captions = clip.tokenize(captions).to(self.device)
        text_features = self.predictor.encode_text(captions)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    def encode_image(self, image: Image.Image):
        image = self.predictor_preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.predictor.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    def get_face(self, bgr):
        faces = DeepFace.extract_faces(bgr, detector_backend='retinaface', expand_percentage=10, normalize_face=False, enforce_detection=False)

        if len(faces) == 1:
            face = Image.fromarray(faces[0]['face'].astype('uint8'), 'RGB')

            return face
        else:
            return None

    def get_human(self, bgr):
        detections = self.human_detector(bgr)[0].boxes.data.cpu().tolist()
        human_detections = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf, cid in detections if cid == 0]

        if len(human_detections) == 1:
            x1, y1, x2, y2 = human_detections[0]

            human = bgr[round(y1):round(y2), round(x1):round(x2)]
            human = cv2.cvtColor(human, cv2.COLOR_BGR2RGB)
            human = Image.fromarray(human.astype('uint8'), 'RGB')

            return human
        else:
            return None

    def predict_facial_beauty(self, bgr, captions=None):
        face = self.get_face(bgr)

        if face is not None:
            image_features = self.encode_image(face)

            if captions is not None:
                text_features = self.encode_captions(captions)
            else:
                text_features = self.text_features

            return self.predict_scores(image_features, text_features)[0][0], face
        else:
            return None, face

    def predict_physical_beauty(self, bgr, captions=None):
        human = self.get_human(bgr)

        if human is not None:
            image_features = self.encode_image(human)

            if captions is not None:
                text_features = self.encode_captions(captions)
            else:
                text_features = self.text_features

            return self.predict_scores(image_features, text_features)[0][0], human
        else:
            return None, human


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AttractiveMeter")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to image"
    )

    args = parser.parse_args()
    image_path = args.image_path

    if not Path(image_path).exists():
        sys.exit(f"File {image_path} does not exist")

    predictor = VisualAttractiveness(["attractive", "unattractive"])

    image = cv2.imread(image_path)

    facial_beauty_score, face = predictor.predict_facial_beauty(image)
    human_beauty_score, human = predictor.predict_physical_beauty(image)

    print(f'FB: {facial_beauty_score}, HB: {human_beauty_score}')
    face.show()
    human.show()
