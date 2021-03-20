from torch import nn
import torch.nn.functional as F

class feature_extractor(nn.Module):

    def __init__(self, backbone):
        super(feature_extractor, self).__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])


    def forward(self, x):
        feats = self.backbone(x)  
        # print(feats.shape)      
        output = F.max_pool2d(feats, kernel_size=feats.size()[2:])
        # print(output.shape)
        output = output.squeeze()
        # print(output.shape)

        return output
