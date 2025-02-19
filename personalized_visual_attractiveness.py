import os.path

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import visual_attractiveness

device = torch.device("cuda" if torch.cuda.is_available() else "mps")


def show_rgb_image(rgb):
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()


def load_images(predictor, folder_name, file_names):
    latents = []

    for filename in file_names:
        img_path = os.path.join(folder_name, filename)
        bgr = cv2.imread(img_path)

        show_rgb_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if bgr is not None:
            human = predictor.get_human(bgr)
            latent = predictor.encode_image(human)

            latents.append(latent)

    latents = torch.cat(latents, dim=0)

    return latents


def test(predictor, train_file_names, scores_train, test_file_names, folder_name):
    captions = ["attractive", "unattractive"]
    encoded_captions = predictor.encode_captions(captions)
    text_features = nn.Parameter(torch.tensor(encoded_captions, dtype=torch.float32, requires_grad=True))

    image_features_train = load_images(predictor, folder_name, train_file_names)
    image_features_train = torch.tensor(image_features_train, dtype=torch.float32).to(device)

    scores_train = torch.tensor(scores_train, dtype=torch.float32).to(device)

    image_features_test = load_images(predictor, folder_name, test_file_names)
    image_features_test = torch.tensor(image_features_test, dtype=torch.float32).to(device)

    optimizer = optim.Adam([text_features], lr=1e-4)
    mseloss = nn.MSELoss()

    # BEFORE

    predicted_scores_before = predictor.predict_scores(image_features_test, text_features)

    # FINETUNING

    stop_epochs = 5

    stop_counter = 0
    previous_loss = 100

    while stop_counter < stop_epochs:
        predicted_scores = predictor.predict_scores(image_features_train, text_features)

        loss = mseloss(predicted_scores, scores_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < previous_loss:
            stop_counter = 0

            previous_loss = loss.item()
        else:
            stop_counter += 1

        print(f"Loss: {loss.item()}")

    # AFTER

    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    predicted_scores_after = predictor.predict_scores(image_features_test, text_features)

    print('')
    print('BEFORE:')
    print(predicted_scores_before)
    print('AFTER:')
    print(predicted_scores_after)


def test_hair_color(predictor, folder_name):
    train_file_names = ['0dd3bdd9f75242d3c8c34eea5c3eae7c.jpg', '0dd7eab34395055a8547b26df6b06b06.jpg', '0e136998854dd3fda68fd07375c2bc1d.jpg', '1b0f1a8ec2e3ba5f4c68936f83e50fde.jpg', '0ff7c3c133b3687a4975bb878aabd02d.jpg']
    scores_train = [[10.0, 0.0], [10.0, 0.0], [8.0, 2.0], [6.0, 4.0], [5.0, 5.0]]

    test_file_names = ['0fbbdcc47bc8356cbd3186ffb880b21e.jpg', '0fda9d1e0e1aec4f5cc18b03020a5371.jpg', '0fb70e487e1036ef7d2a288a4b034601.jpg', '0fb803ceb5c6f3d1c95a5f4efb3a2108.jpg']

    test(predictor, train_file_names, scores_train, test_file_names, folder_name)


def test_skin_color(predictor, folder_name):
    train_file_names = ['0f273338b78f9d2014ae8e2a0659c0e4.jpg', '2c14baf35ae5512ab91fc2546515b754.jpg', '2fc37c24dcfd85da4bea13fcd3cff0e5.jpg']
    scores_train = [[8.0, 2.0], [7.0, 3.0], [10.0, 0.0]]

    test_file_names = ['6d07c19ad563a6d92c771e1141b1c7ca.jpg', '06f87355f734a1208f3fa83e668354a6.jpg']

    test(predictor, train_file_names, scores_train, test_file_names, folder_name)


if __name__ == "__main__":
    predictor = visual_attractiveness.VisualAttractiveness()

    folder_name = '/Users/alexanderfuchs/Desktop/WS_24_25/TinderBotz/data_man2woman/geomatches/images'

    test_skin_color(predictor, folder_name)

