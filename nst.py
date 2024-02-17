import os
import argparse
import torch
import utils.utils as utils
import numpy as np
import model.vgg as vgg
from torchvision import transforms
import cv2 as cv
from utils.utils import IMAGENET_STD_NEUTRAL, IMAGENET_MEAN_255


def reconstruct(config):
    """
    overview:
    -get style and content images, and initialize white noize as output
    -get feature maps from vgg model
    -compute losses
    -update output

    :param config:
    :return: None
    """

    contentDir = config['content_dir']
    styleDir = config['style_dir']
    outputDir = config['output_dir']
    content_img_path = os.path.join(contentDir, config['content_img_name'])
    style_img_path = os.path.join(styleDir, config['style_img_name'])
    height = config['height']
    alpha = config['content_weight']
    beta = config['style_weight']
    gamma = config['var_weight']

    device = torch.device(config['device'])

    # TODO: make sure the two are of the same size
    content_img = utils.prepare_img(content_img_path, height, device)
    print(content_img.shape)
    width = content_img.shape[3]
    style_img = utils.prepare_sty(style_img_path, height, width, device)
    print(style_img.shape)
    # TODO: tune the value 90
    output = np.random.uniform(-50., 50., content_img.shape).astype(np.float32)
    output = torch.from_numpy(output).float().to(device)
    output = torch.autograd.Variable(output, requires_grad=True)

    model = vgg.Vgg19(use_relu=True)
    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names
    model.to(device).eval()
    content_feature_maps = model(content_img)
    style_feature_maps = model(style_img)

    target_feature = content_feature_maps[content_feature_maps_index]
    target_feature = target_feature.squeeze(0)
    target_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in enumerate(style_feature_maps) if
                                   i in style_feature_maps_indices]
    if config['optimizer'] == 'lbfgs':
        epochs = 100
        optimizer = torch.optim.LBFGS((output,), max_iter=epochs, line_search_fn='strong_wolfe')
        count = 0

        def closure():
            nonlocal count
            count += 1
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            output_feature = model(output)[content_feature_maps_index]
            loss = torch.nn.MSELoss(reduction='mean')(output_feature.squeeze(0), target_feature) * alpha
            current_set_of_feature_maps = model(output)
            current_style_representation = [utils.gram_matrix(fmaps) for i, fmaps in
                                            enumerate(current_set_of_feature_maps) if
                                            i in style_feature_maps_indices]
            for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                loss += (beta / len(target_style_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0],
                                                                                                       gram_hat[0])

            loss += gamma * utils.variation_loss(output)

            if loss.requires_grad:
                loss.backward()
            with torch.no_grad():
                filename = str(count) + 'output.jpg'
                out_img = output.squeeze(axis=0).to('cpu').detach().numpy()
                out_img = np.moveaxis(out_img, 0, 2)
                dump_img = np.copy(out_img)
                dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
                dump_img = np.clip(dump_img, 0, 255).astype('uint8')
                cv.imwrite(os.path.join(outputDir, filename), np.array(dump_img)[:, :, ::-1])
                print(loss)
                return loss

        optimizer.step(closure)


if __name__ == "__main__":
    resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_dir = os.path.join(resource_dir, 'content_image')
    style_dir = os.path.join(resource_dir, 'style_image')
    output_dir = os.path.join(resource_dir, 'output')

    parser = argparse.ArgumentParser()
    parser.add_argument("--content_weight", type=float, default=100)
    parser.add_argument("--style_weight", type=float, default=0.04)
    parser.add_argument("--var_weight", type=float, default=0.001)
    parser.add_argument("--content_img_name", type=str, default="tubingen.png")
    parser.add_argument("--style_img_name", type=str, default="vg_starry_night.jpg")
    parser.add_argument("--height", type=int, help="height of reconstruction", default=500)
    parser.add_argument("--optimizer", type=str, choices=['adam', 'lbfgs'], default='lbfgs')
    parser.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()

    configure = dict()
    configure['resource_dir'] = resource_dir
    configure['content_dir'] = content_dir
    configure['style_dir'] = style_dir
    configure['output_dir'] = output_dir
    for arg in vars(args):
        configure[arg] = getattr(args, arg)

    reconstruct(configure)
