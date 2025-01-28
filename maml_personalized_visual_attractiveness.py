import random

import torch
import torch.nn as nn
import torch.optim as optim

import dataset_mebeauty as mebeauty
import visual_attractiveness

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def get_batch(predictor, raters_with_scores, image_path_to_latent, batch_size, k):
    batch = []
    for index in range(batch_size):
        rater = random.choice(list(raters_with_scores.keys()))

        image_paths_to_scores = list(raters_with_scores[rater].items())
        #image_paths_to_scores = random.choices(image_paths_to_scores, k=3 * k)

        image_paths = list(zip(*image_paths_to_scores))[0]

        #image_path_to_latent = mebeauty.load_image_path_to_latent(predictor, image_paths)

        latent_to_score = [(image_path_to_latent[image_path], score) for image_path, score in image_paths_to_scores if image_path in list(image_path_to_latent.keys())]

        if len(latent_to_score) >= 2 * k:
            latents, scores = list(zip(*latent_to_score))

            image_features_support = torch.cat(latents[:k], dim=0)
            image_features_support = torch.tensor(image_features_support, dtype=torch.float32)

            scores_support = torch.tensor(scores[:k], dtype=torch.float32).to(device)
            scores_support = scores_support.unsqueeze(-1)
            scores_support = torch.cat([scores_support, 10 - scores_support], dim=1)

            image_features_query = torch.cat(latents[k:], dim=0)
            image_features_query = torch.tensor(image_features_query, dtype=torch.float32)

            scores_query = torch.tensor(scores[k:], dtype=torch.float32).to(device)
            scores_query = scores_query.unsqueeze(-1)
            scores_query = torch.cat([scores_query, 10 - scores_query], dim=1)

            batch.append((image_features_support, scores_support, image_features_query, scores_query))

    return batch


def train_maml(number_of_batches, batch_size, k):
    #predictor_old = visual_attractiveness.VisualAttractiveness()
    predictor = visual_attractiveness.VisualAttractiveness()

    raters_with_scores = mebeauty.load_rater_to_personalized_scores()
    raters_with_scores = {rater: scores for rater, scores in raters_with_scores.items() if len(scores) >= 3 * k}

    image_path_to_latent = mebeauty.load_image_path_to_latent(predictor)

    text_features_old = torch.load("weights/text_features_mebeauty_100k.pth")
    text_features = nn.Parameter(text_features_old, requires_grad=True)

    outer_optimizer = optim.Adam([
        #{"params": predictor.predictor.parameters()},
        {"params": [text_features]},
    ], lr=1e-3)

    loss_diff = []
    for batch_index in range(number_of_batches):
        loss_before = 0
        loss_after = 0

        batch = get_batch(predictor, raters_with_scores, image_path_to_latent, batch_size, k)

        if len(batch) < batch_size / 2:
            continue

        for image_features_support, scores_support, image_features_query, scores_query in batch:
            #clone_predictor = visual_attractiveness.VisualAttractiveness(predictor_to_clone=predictor)
            clone_text_features = nn.Parameter(text_features.clone().detach(), requires_grad=True)

            inner_optimizer = optim.Adam([
                #{"params": clone_predictor.predictor.parameters()},
                {"params": [clone_text_features]},
            ], lr=1e-4)
            inner_mse_loss = nn.MSELoss()

            predicted_scores = predictor.predict_scores(image_features_query, text_features_old)
            loss_before = loss_before + inner_mse_loss(predicted_scores, scores_query)

            for i in range(3):
                predicted_scores = predictor.predict_scores(image_features_support, clone_text_features)
                inner_loss = inner_mse_loss(predicted_scores, scores_support)

                inner_optimizer.zero_grad()
                inner_loss.backward()
                inner_optimizer.step()

            predicted_scores = predictor.predict_scores(image_features_query, clone_text_features)
            loss_after = loss_after + inner_mse_loss(predicted_scores, scores_query)

        loss_before = loss_before / len(batch)
        loss_after = loss_after / len(batch)

        outer_optimizer.zero_grad()
        loss_after.backward()
        outer_optimizer.step()

        loss_diff.append(loss_before - loss_after)

        print(f"Loss diff: {torch.mean(torch.tensor(loss_diff)):.4f}")

    torch.save(text_features, "weights/text_features_mebeauty_maml_k10_50k.pth")
    print("Text Features Mebeauty Maml saved!")

    #torch.save(predictor.predictor.state_dict(), "weights/predictor_mebeauty_maml_k10_50k.pth")
    #print("Predictor Mebeauty Maml saved!")


if __name__ == "__main__":
    train_maml(80, 16, 10)