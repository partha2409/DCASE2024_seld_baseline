import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class VideoFeatures(nn.Module):
    def __init__(self):
        super(VideoFeatures, self).__init__()

        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.backbone = torch.nn.Sequential(*(list(self.model.children())[:-2]))
        self.backbone.eval()
        self.preprocess = self.weights.transforms()

    def forward(self, images):
        with torch.no_grad():
            preprocessed_images = [self.preprocess(image) for image in images]
            preprocessed_images = torch.stack(preprocessed_images, dim=0)
            vid_features = self.backbone(preprocessed_images)
            vid_features = torch.mean(vid_features, dim=1)
            return vid_features


