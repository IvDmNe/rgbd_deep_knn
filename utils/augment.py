import torch
from torchvision.transforms import Lambda, Compose, Resize, ToTensor
from PIL import Image

def convert_to_tensor(images, shape=(224,224)):


    # check if images are grayscale
    if len(images[0].shape) == 2:
        trs = Compose([Resize(shape),
                ToTensor(),
                Lambda(lambda tens: torch.cat([tens, tens, tens])),
                
        ])

    else:
        trs = Compose([Resize(shape),
                ToTensor(),
        ])

    res_images = []
    for image in images:
        image = Image.fromarray(image)
        tens_image = trs(image)
        res_images.append(tens_image)

    res_images = torch.stack(res_images)
    return res_images
