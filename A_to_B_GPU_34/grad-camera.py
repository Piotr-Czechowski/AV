import os
import cv2
import numpy as np
import torch
# from torchvision import models
from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST
from ACTIONS import ACTIONS


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    from collections import namedtuple

    Args = namedtuple('Args', ['device', 'image_path', 'model_path', 'speed', 'method', 'aug_smooth',
                               'eigen_smooth', 'output_dir'])
    args = Args(device='cuda',
                image_path='ep_100_step_98.jpeg',
                model_path=r"C:\Users\barte\Downloads\synchr_200_semantic_camera_7_2_img_speed.pth",
                speed= torch.Tensor([0.5]),
                method='gradcam',
                aug_smooth=True,
                eigen_smooth=True,
                output_dir=r'.')

    # Choose nets with speed or without
    if args.speed is None:
        from nets.a2c_no_speed import DiscreteActor
    else:
        from nets.a2c import DiscreteActor

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "fem": FEM,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM,
        'shapleycam': ShapleyCAM,
        'finercam': FinerCAM
    }

    if args.device == 'hpu':
        import habana_frameworks.torch.core as htcore

    state_shape = [200, 200, 3]
    actions_shape = len(ACTIONS.ACTIONS_NAMES)
    model = DiscreteActor(state_shape, actions_shape, args.device).to(args.device)
    model_values = torch.load(args.model_path, map_location=args.device)['actor']
    # Change keys like below if models does not fully match
    # model_values['scalar_layer.0.weight'] = model_values.pop('speed_layer1.0.weight')
    # model_values['scalar_layer.0.bias'] = model_values.pop('speed_layer1.0.bias')
    # model_values['logits.weight'] = model_values.pop('actor.weight')
    # model_values['logits.bias'] = model_values.pop('actor.bias')
    # ######
    model.load_state_dict(model_values)

    # # Choose the target layer you want to compute the visualization for.
    # # Usually this will be the last convolutional layer in the model.
    # # Some common choices can be:
    # # Resnet18 and 50: model.layer4
    # # VGG, densenet161: model.features[-1]
    # # mnasnet1_0: model.layers[-1]
    # # You can print the model to help chose the layer
    # # You can pass a list with several target layers,
    # # in that case the CAMs will be computed per layer and then aggregated.
    # # You can also try selecting all layers of a certain type, with e.g:
    # # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # # find_layer_types_recursive(model, [torch.nn.ReLU])

    target_layers = [model.layer3]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(args.device)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputReST(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth,
                            speed=args.speed)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    os.makedirs(args.output_dir, exist_ok=True)
    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    cv2.imwrite(cam_output_path, cam_image)

    print(' --- DONE --- ')

    # """ Additional Images """
    # gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
    # gb = gb_model(input_tensor, target_category=None)
    #
    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)
    #
    # gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
    # cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')
    #
    # cv2.imwrite(gb_output_path, gb)
    # cv2.imwrite(cam_gb_output_path, cam_gb)
