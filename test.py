import torch

from network import AvatarNet
from utils import imload, imload_web, imsave, maskload


def network_test(args):
    # set device
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')
    
    # load check point
    check_point = torch.load(args.check_point)

    # load network
    network = AvatarNet(args.layers)
    network.load_state_dict(check_point['state_dict'])
    network = network.to(device)

    # load target images
    content_img = imload(args.content, args.imsize, args.cropsize).to(device)
    style_imgs = [imload(style, args.imsize, args.cropsize, args.cencrop).to(device) for style in args.style]
    masks = None
    if args.mask:
        masks = [maskload(mask).to(device) for mask in args.mask]

    # stylize image
    with torch.no_grad():
        stylized_img =  network(content_img, style_imgs, args.style_strength, args.patch_size, args.patch_stride,
                masks, args.interpolation_weights, False)

    imsave(stylized_img, 'stylized_image.jpg')

    return None


def network_test_web(device, network, imsize, style_strength, patch_size, patch_stride, interpolation_weights, content, styles, cropsize, cencrop):# TODO remove args

    # TODO switch below
    # load target images
    content_img = imload_web(content, imsize, cropsize).to(device)
    style_imgs = [imload_web(style, imsize, cropsize, cencrop).to(device) for style in styles]
    masks = None
    # if False:#args.mask:
    #     masks = [maskload(mask).to(device) for mask in args.mask]

    # stylize image
    with torch.no_grad():
        stylized_img =  network(content_img, style_imgs, style_strength, patch_size, patch_stride,
                masks, interpolation_weights, False)

    # TODO update this to be exporting
    # imsave(stylized_img, 'stylized_image.jpg')
    print(type(stylized_img))

    return None
