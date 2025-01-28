import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import dataset_mebeauty as mebeauty
import visual_attractiveness

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def finetune_textual_embeddings(number_of_epochs, batch_size):
    predictor = visual_attractiveness.VisualAttractiveness()

    text_features = torch.load("weights/text_features_mebeauty_100k.pth")
    text_features = nn.Parameter(torch.tensor(text_features, dtype=torch.float32, requires_grad=True))

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam([text_features], lr=1e-3)

    image_path_to_score = mebeauty.load_image_path_to_score()
    image_path_to_score = list(image_path_to_score.items())
    image_path_to_score = DataLoader(image_path_to_score, batch_size=batch_size, collate_fn=lambda batch: batch, shuffle=True)

    losses = []
    for epoch_index in range(number_of_epochs):
        for batch in image_path_to_score:
            image_paths = list(zip(*batch))[0]

            image_path_to_latent = mebeauty.load_image_path_to_latent(predictor, image_paths)

            latent_to_score = [(image_path_to_latent[image_path], score) for image_path, score in batch if image_path in list(image_path_to_latent.keys())]

            latents, scores = list(zip(*latent_to_score))

            image_features = torch.cat(latents, dim=0)
            image_features = torch.tensor(image_features, dtype=torch.float32)

            predicted_scores = predictor.predict_scores(image_features, text_features)

            scores = torch.tensor(scores, dtype=torch.float32).to(device)
            scores = scores.unsqueeze(-1)
            scores = torch.cat([scores, 10 - scores], dim=1)

            loss = mse_loss(predicted_scores, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            print(f"Loss: {torch.mean(torch.tensor(losses)):.4f}")

        print(f"Epoch [{epoch_index + 1}], Loss: {torch.mean(torch.tensor(losses)):.4f}")

    torch.save(text_features, "weights/text_features_mebeauty_200k.pth")
    print("Text Features Mebeauty saved!")


if __name__ == "__main__":
    finetune_textual_embeddings(20, 64)
