import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import dataset_mebeauty as mebeauty
import visual_attractiveness

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def train_1st_stage(number_of_epochs, batch_size):
    predictor = visual_attractiveness.VisualAttractiveness()

    text_features = torch.load("weights/text_features_mebeauty_50k.pth")
    text_features = nn.Parameter(torch.tensor(text_features, dtype=torch.float32, requires_grad=True)).to(device)

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam([text_features], lr=1e-4)

    image_path_to_score = mebeauty.load_image_path_to_score()
    image_path_to_score = list(image_path_to_score.items())
    image_path_to_score = DataLoader(image_path_to_score, batch_size=batch_size, collate_fn=lambda batch: batch, shuffle=True)

    losses = []
    for epoch_index in range(number_of_epochs):
        for batch in image_path_to_score:
            image_paths = list(zip(*batch))[0]

            image_path_to_latent = mebeauty.load_image_path_to_latent(image_paths)

            latent_to_score = [(image_path_to_latent[image_path], score) for image_path, score in batch if image_path in list(image_path_to_latent.keys())]
            latents, scores = list(zip(*latent_to_score))

            image_features = torch.cat(latents, dim=0)
            image_features = torch.tensor(image_features, dtype=torch.float32).to(device)

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

    torch.save(text_features, "weights/text_features_mebeauty_100k.pth")
    print("Text Features Mebeauty saved!")


def batch_2nd_stage(raters_with_scores, image_path_to_latent, k, batch_size):
    images_support_batch = []
    images_query_batch = []
    scores_support_batch = []
    scores_query_batch = []
    support_batch = []

    for index in range(batch_size):
        rater = random.choice(list(raters_with_scores.keys()))

        image_paths_to_scores = raters_with_scores[rater]

        latent_to_score = [(image_path_to_latent[image_path], score) for image_path, score in image_paths_to_scores if image_path in list(image_path_to_latent.keys())]
        latent_to_score = random.choices(latent_to_score, k=k * 2)

        if len(latent_to_score) == k * 2:
            latents, scores = list(zip(*latent_to_score))

            images_support = torch.cat(latents[:k], dim=0)
            images_support = torch.tensor(images_support, dtype=torch.float32).to(device)
            images_support_batch.append(images_support)

            images_query = torch.cat(latents[k:], dim=0)
            images_query = torch.tensor(images_query, dtype=torch.float32).to(device)
            images_query_batch.append(images_query)

            scores_support = torch.tensor(scores[:k], dtype=torch.float32).to(device)
            scores_support = scores_support.unsqueeze(-1)
            scores_support_batch.append(scores_support)

            scores_query = torch.tensor(scores[k:], dtype=torch.float32).to(device)
            scores_query = scores_query.unsqueeze(-1)
            scores_query = torch.cat([scores_query, 10 - scores_query], dim=1)
            scores_query_batch.append(scores_query)

            support = torch.cat([scores_support, images_support], dim=1)
            support_batch.append(support)

    support_batch = torch.stack(support_batch)
    images_query_batch = torch.stack(images_query_batch)
    scores_query_batch = torch.stack(scores_query_batch)

    return support_batch, images_query_batch, scores_query_batch


def train_2nd_stage(k, number_of_batches, batch_size):
    predictor = visual_attractiveness.VisualAttractiveness()
    generator = visual_attractiveness.TextFeaturesGenerator(k)

    text_features = torch.load("weights/text_features_mebeauty_100k.pth")
    text_features = torch.tensor(text_features, dtype=torch.float32, requires_grad=True).to(device)

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=1e-3)

    raters_with_scores = mebeauty.load_rater_to_personalized_scores()
    raters_with_scores = {rater: scores for rater, scores in raters_with_scores.items() if len(scores) >= k * 3}

    image_path_to_latent = mebeauty.load_image_path_to_latent()

    losses = []
    for batch_index in range(number_of_batches):
        support, images_query, scores_query = batch_2nd_stage(raters_with_scores, image_path_to_latent, k, batch_size)

        predicted_delta_text_features = generator(support)

        predicted_text_features = text_features + predicted_delta_text_features

        predicted_scores = predictor.predict_scores(images_query, predicted_text_features)

        loss = mse_loss(predicted_scores, scores_query)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f"Loss: {torch.mean(torch.tensor(losses)):.4f}")

        if (batch_index + 1) % 40 == 0:
            print(f"Epoch [{(batch_index + 1) // 40}], Loss: {torch.mean(torch.tensor(losses)):.4f}")


    torch.save(generator.state_dict(), "weights/generator_mebeauty.pth")
    print("Generator parameters saved!")


if __name__ == "__main__":
    #train_1st_stage(20, 64)

    train_2nd_stage(3, 10_000, 32)