# INSPIRATION: https://github.com/pengzhiliang/MAE-pytorch/blob/main/run_mae_vis.py
# import torch

# from torchvision import datasets, transforms
# from masking_generator import RandomMaskingGenerator


# class DataAugmentationForMAE(object):
#     def __init__(self, args):
#         imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
#         # mean = (
#         #     IMAGENET_INCEPTION_MEAN
#         #     if not imagenet_default_mean_and_std
#         #     else IMAGENET_DEFAULT_MEAN
#         # )
#         # std = (
#         #     IMAGENET_INCEPTION_STD
#         #     if not imagenet_default_mean_and_std
#         #     else IMAGENET_DEFAULT_STD
#         # )

#         self.transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(args.input_size),
#                 transforms.ToTensor(),
#                 # transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
#             ]
#         )

#         self.masked_position_generator = RandomMaskingGenerator(
#             args.window_size, args.mask_ratio
#         )

#     def __call__(self, image):
#         return self.transform(image), self.masked_position_generator()

#     def __repr__(self):
#         repr = "(DataAugmentationForBEiT,\n"
#         repr += "  transform = %s,\n" % str(self.transform)
#         repr += "  Masked position generator = %s,\n" % str(
#             self.masked_position_generator
#         )
#         repr += ")"
#         return repr
