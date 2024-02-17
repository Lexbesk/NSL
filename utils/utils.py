import numpy as np
import torch
import cv2 as cv
from torchvision import transforms


IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def prepare_img(img_path, height, device):
    print(img_path)
    img = cv.imread(img_path)[:, :, ::-1]
    h, w, _ = img.shape
    width = int(height * (w / h))
    img = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)])

    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)

    return img


def prepare_sty(img_path, h, w, device):
    print(img_path)
    img = cv.imread(img_path)[:, :, ::-1]
    img = cv.resize(img, (w, h), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
        ])

    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)

    return img


def get_grams(features):
    res = []
    for i in range(len(features)):
        x = features[i]
        (b, ch, h, w) = x.size()
        y = x.view(b, ch, w * h)
        features_t = y.transpose(1, 2)
        gram = y.bmm(features_t)
        gram /= 4 * (ch * h * w) ** 2
        res.append(gram)
    return res


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= (ch * h * w)
    return gram


def variation_loss(output):
    var_x = output[:, :, :, :-1] - output[:, :, :, 1:]
    var_y = output[:, :, :-1, :] - output[:, :, 1:, :]
    return torch.sum(torch.abs(var_x)) + torch.sum(torch.abs(var_y))

