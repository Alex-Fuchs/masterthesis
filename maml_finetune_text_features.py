import random

import torch
import torch.nn as nn
import torch.optim as optim

import dataset_mebeauty as mebeauty
import visual_attractiveness

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def get_batch(predictor, raters_with_scores, batch_size, k):
    batch = []
    for index in range(batch_size):
        rater = random.choice(list(raters_with_scores.keys()))

        image_paths_to_scores = list(raters_with_scores[rater].items())
        image_paths_to_scores = random.choices(image_paths_to_scores, k=k + 18)

        image_paths = list(zip(*image_paths_to_scores))[0]

        image_path_to_latent = mebeauty.load_image_path_to_latent(predictor, image_paths)

        latent_to_score = [(image_path_to_latent[image_path], score) for image_path, score in image_paths_to_scores if image_path in list(image_path_to_latent.keys())]

        latents, scores = list(zip(*latent_to_score))

        image_features_support = torch.cat(latents[:k], dim=0)
        image_features_support = torch.tensor(image_features_support)

        scores_support = torch.tensor(scores[:k], dtype=torch.float32).to(device)
        scores_support = scores_support.unsqueeze(-1)
        scores_support = torch.cat([scores_support, 10 - scores_support], dim=1)

        image_features_query = torch.cat(latents[k:], dim=0)
        image_features_query = torch.tensor(image_features_query)

        scores_query = torch.tensor(scores[k:], dtype=torch.float32).to(device)
        scores_query = scores_query.unsqueeze(-1)
        scores_query = torch.cat([scores_query, 10 - scores_query], dim=1)

        batch.append((image_features_support, scores_support, image_features_query, scores_query))

        return batch


def train_maml(number_of_batches, batch_size, k, s):
    predictor = visual_attractiveness.VisualAttractiveness()

    raters_with_scores = mebeauty.load_rater_to_personalized_scores()
    raters_with_scores = {rater: scores for rater, scores in raters_with_scores.items() if len(scores) >= k + 18}

    text_features_old = torch.load(f"weights/text_features_mebeauty_100k.pth")
    text_features = nn.Parameter(text_features_old, requires_grad=True)

    outer_optimizer = optim.Adam([
        {"params": [text_features]},
    ], lr=1e-4)

    losses_before = []
    losses_after = []
    for batch_index in range(number_of_batches):
        loss_before = 0
        loss_after = 0

        batch = get_batch(predictor, raters_with_scores, batch_size, k)

        for image_features_support, scores_support, image_features_query, scores_query in batch:
            clone_text_features = nn.Parameter(text_features.clone().detach(), requires_grad=True)

            inner_optimizer = optim.Adam([
                {"params": [clone_text_features]},
            ], lr=1e-4)
            inner_mse_loss = nn.MSELoss()

            predicted_scores = predictor.predict_scores(image_features_query, text_features_old)
            loss_before = loss_before + inner_mse_loss(predicted_scores, scores_query)

            for i in range(s):
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

        losses_before.append(loss_before)
        losses_after.append(loss_after)

        print(f"Loss: {torch.mean(torch.tensor(losses_after))}, Loss Before: {torch.mean(torch.tensor(losses_before)):.4f}")

    torch.save(text_features, f"weights/text_features_mebeauty_maml_k{k}_s{s}_25k.pth")
    print(f"Text Features Mebeauty Maml k={k} s={s} saved!")


def test_maml(number_of_batches, batch_size, k, s):
    predictor = visual_attractiveness.VisualAttractiveness()

    raters_with_scores = mebeauty.load_rater_to_personalized_scores()
    raters_with_scores = {rater: scores for rater, scores in raters_with_scores.items() if len(scores) >= k + 18}

    text_features_old = torch.load(f"weights/text_features_mebeauty_100k.pth")

    text_features = torch.load(f"weights/text_features_mebeauty_maml_k{k}_s{s}_25k.pth")
    text_features = nn.Parameter(text_features, requires_grad=True)

    diff_before = []
    diff_after = []
    for batch_index in range(number_of_batches):

        batch = get_batch(predictor, raters_with_scores, batch_size, k)

        for image_features_support, scores_support, image_features_query, scores_query in batch:
            clone_text_features = nn.Parameter(text_features.clone().detach(), requires_grad=True)

            inner_optimizer = optim.Adam([
                {"params": [clone_text_features]},
            ], lr=1e-4)
            inner_mse_loss = nn.MSELoss()

            predicted_scores = predictor.predict_scores(image_features_query, text_features_old)
            diff_before.append(torch.mean(torch.abs(predicted_scores[:, 0] - scores_query[:, 0])))

            for i in range(s):
                predicted_scores = predictor.predict_scores(image_features_support, clone_text_features)
                inner_loss = inner_mse_loss(predicted_scores, scores_support)

                inner_optimizer.zero_grad()
                inner_loss.backward()
                inner_optimizer.step()

            predicted_scores = predictor.predict_scores(image_features_query, clone_text_features)
            diff_after.append(torch.mean(torch.abs(predicted_scores[:, 0] - scores_query[:, 0])))

    print(f"MAE k={k} s={s}: {torch.mean(torch.stack(diff_after))}, MAE Before: {torch.mean(torch.stack(diff_before)):.4f}")


if __name__ == "__main__":
    print('Training...')
    train_maml(85, 16, 3, 3)
    train_maml(85, 16, 3, 6)
    train_maml(85, 16, 3, 9)
    train_maml(75, 16, 5, 3)
    train_maml(75, 16, 5, 6)
    train_maml(75, 16, 5, 9)
    train_maml(61, 16, 10, 3)
    train_maml(61, 16, 10, 6)
    train_maml(61, 16, 10, 9)

    '''print('Testing...')
    test_maml(32, 16, 3, 3)
    test_maml(32, 16, 3, 6)
    test_maml(32, 16, 3, 9)
    test_maml(32, 16, 5, 3)
    test_maml(32, 16, 5, 6)
    test_maml(32, 16, 5, 9)
    test_maml(32, 16, 10, 3)
    test_maml(32, 16, 10, 6)
    test_maml(32, 16, 10, 9)'''
