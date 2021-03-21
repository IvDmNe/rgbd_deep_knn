import torch
from torchvision.transforms import Lambda, Compose, Resize, ToTensor

def convert_to_tensor(images, shape=(224,224)):


    # check if images are grayscale
    if len(images[0].shape) == 2:
        trs = Compose([ToTensor(),
                Lambda(lambda tens: torch.cat([tens, tens, tens])),
                Resize(shape)
        ])

    else:
        trs = Compose([ToTensor(),
                Resize(shape)
        ])

    res_images = []
    for image in images:
        tens_image = trs(image)
        # print(image.shape)
        # print(tens_image.shape)
        res_images.append(tens_image)

    res_images = torch.stack(res_images)
    # print(res_images.shape)
    return res_images
