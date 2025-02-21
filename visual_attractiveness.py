import argparse
import sys

from pathlib import Path

import cv2
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn

from transformers import CLIPProcessor, CLIPModel
from deepface import DeepFace
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


class VisualAttractiveness(nn.Module):

    def __init__(self, captions=None, text_features=None):
        super().__init__()

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            device_map=device,
            torch_dtype=torch.float32
        )

        self.preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.human_detector = YOLO("yolov8n.pt")

        if text_features is None:
            if captions is None:
                self.text_features = self.encode_captions(["attractive", "unattractive"])
            else:
                self.text_features = self.encode_captions(captions)
        else:
            self.text_features = text_features

    def predict_scores(self, image_features, text_features=None):
        if text_features is None:
            text_features = self.text_features

        text_features = text_features.to(device)
        image_features = image_features.to(device)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        scores = logits_per_image.softmax(dim=-1) * 10

        return scores

    def encode_captions(self, captions):
        text_inputs = self.preprocessor(text=captions, return_tensors="pt", padding=True).to(device)
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

    def encode_image(self, image):
        image_inputs = self.preprocessor(images=image, return_tensors="pt").to(device)
        image_features = self.model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    def get_face(self, bgr):
        faces = DeepFace.extract_faces(bgr, detector_backend='retinaface', expand_percentage=10, normalize_face=False, enforce_detection=False)

        if len(faces) == 1:
            face = faces[0]['face']
            face = Image.fromarray(face.astype('uint8'), 'RGB')

            return face
        else:
            return None

    def get_human(self, bgr):
        detections = self.human_detector(bgr, verbose=False)[0].boxes.data.cpu().tolist()
        human_detections = [(x1, y1, x2, y2) for x1, y1, x2, y2, conf, cid in detections if cid == 0]

        if len(human_detections) == 1:
            x1, y1, x2, y2 = human_detections[0]

            human = bgr[round(y1):round(y2), round(x1):round(x2)]
            human = cv2.cvtColor(human, cv2.COLOR_BGR2RGB)
            human = Image.fromarray(human.astype('uint8'), 'RGB')

            return human
        else:
            return None

    @torch.no_grad()
    def explainable_heatmap_rise(self, image, text_features=None, num_masks=10000, mask_size=8, batch_size=264):
        image_inputs = self.preprocessor(images=image, return_tensors="pt").to(device)

        saliency_map = torch.zeros(224, 224).to(device)
        saliency_map_count = torch.zeros(224, 224).to(device)

        for m in range(num_masks // batch_size + 1):
            masks = []
            for b in range(batch_size):
                mask = np.random.choice([0, 1], size=(mask_size, mask_size), p=[0.5, 0.5])
                mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device)
                mask = torch.nn.functional.interpolate(mask, size=(224, 224), mode="nearest")

                masks.append(mask)

            masks = torch.cat(masks, dim=0)

            image_inputs_masked = image_inputs['pixel_values'] * masks

            image_features_masked = self.model.get_image_features(image_inputs_masked)
            image_features_masked = image_features_masked / image_features_masked.norm(dim=1, keepdim=True)

            scores = self.predict_scores(image_features_masked, text_features)[:, 0]

            masks = masks.squeeze()
            scores = scores.unsqueeze(-1).unsqueeze(-1)

            saliency_map += torch.sum(masks * scores, dim=0)
            saliency_map_count += torch.sum(masks, dim=0)

        saliency_map = saliency_map / saliency_map_count
        saliency_map = saliency_map.cpu().numpy()

        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        saliency_map = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.resize(np.array(image), (224, 224)))
        plt.axis("off")
        plt.title("Original image (224x224)")

        plt.subplot(1, 2, 2)
        plt.imshow(saliency_map)
        plt.axis("off")
        plt.title("Saliency Map (RISE)")
        plt.show()

    '''def explainable_heatmap_gradient(self, rgb, text_features=None):
        image_inputs = self.preprocess(images=image, return_tensors="pt").to(device)
        image_inputs.requires_grad = True

        image_features = self.predictor.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        scores = self.predict_scores(image_features, text_features)

        loss = -scores[0, 0]
        loss.backward()

        gradients = image_inputs.grad.abs().cpu().detach().numpy()[0]
        gradients = np.mean(gradients, 0)

        saliency_map = (gradients - gradients.min()) / (gradients.max() - gradients.min())
        saliency_map = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)

        np_image = cv2.resize(np.array(rgb), (saliency_map.shape[1], saliency_map.shape[0]))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(np_image)
        plt.axis("off")
        plt.title("Original image (224x224)")

        plt.subplot(1, 2, 2)
        plt.imshow(saliency_map)
        plt.axis("off")
        plt.title("Saliency Map (Gradient)")
        plt.show()

        return scores'''

    @torch.no_grad()
    def explainable_heatmap_facial(self, bgr, text_features=None):
        face = self.get_face(bgr)

        if face is not None:
            self.explainable_heatmap_rise(face, text_features)

    @torch.no_grad()
    def explainable_heatmap_physical(self, bgr, text_features=None):
        human = self.get_human(bgr)

        if human is not None:
            self.explainable_heatmap_rise(human, text_features)

    @torch.no_grad()
    def predict_facial_beauty(self, bgr, text_features=None):
        face = self.get_face(bgr)

        if face is not None:
            image_features = self.encode_image(face)

            return self.predict_scores(image_features, text_features)[0][0].item(), face
        else:
            return None, face

    @torch.no_grad()
    def predict_physical_beauty(self, bgr, text_features=None):
        human = self.get_human(bgr)

        if human is not None:
            image_features = self.encode_image(human)

            return self.predict_scores(image_features, text_features)[0][0].item(), human
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

    predictor = VisualAttractiveness()

    image = cv2.imread(image_path)

    score, _ = predictor.predict_physical_beauty(image)

    print(score)
