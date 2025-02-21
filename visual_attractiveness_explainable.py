import re

import cv2

import numpy as np

import torch
import torch.nn as nn

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import visual_attractiveness
import dataset_mebeauty as mebauty

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


class VisualAttractivenessExplainable(nn.Module):

    def __init__(self, query=None):
        super().__init__()

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            max_pixels=1000 * 1000
        )

        if query is None:
            self.query = "Give an attractiveness rating for the person in the image in following manner: Attractiveness: 0 to 10. Also give a description afterwards."
        else:
            self.query = query

    def predict_physical_beauty_description(self, image_path, query=None):
        if query is None:
            query = self.query

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{image_path}",
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        inputs = inputs.to(device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text[0]

    def predict_physical_beauty(self, image_path, query=None):
        description = self.predict_physical_beauty_description(image_path, query)

        occurences = re.findall(r"Attractiveness: \d+", description)

        if len(occurences) == 1:
            return float(occurences[0][16:])
        else:
            return None


def compare_to_old(predictor_old, predictor_new, image_path_to_score, k=100):
    avg_body = []
    for image_path, score in list(image_path_to_score.items())[:k]:
        human_beauty_score_new = predictor_new.predict_physical_beauty(image_path)
        human_beauty_score_old, _ = predictor_old.predict_physical_beauty(cv2.imread(image_path))

        if human_beauty_score_new is not None and human_beauty_score_old is not None:
            avg_body.append(abs(human_beauty_score_new - human_beauty_score_old))

    print(np.mean(np.array(avg_body)))


if __name__ == "__main__":
    predictor_old = visual_attractiveness.VisualAttractiveness()
    predictor_new = VisualAttractivenessExplainable()

    compare_to_old(predictor_old, predictor_new, mebauty.load_image_path_to_score())
