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
            max_batch_size = 1000
            iter = (len(preprocessed_images) - 1) // max_batch_size + 1
            vid_features_part_list = []
            for i in range(iter):
                preprocessed_images_part = torch.stack(preprocessed_images[i * max_batch_size: (i + 1) * max_batch_size], dim = 0)
                vid_features_part = self.backbone(preprocessed_images_part)
                vid_features_part = torch.mean(vid_features_part, dim=1)
                vid_features_part_list.append(vid_features_part)
            vid_features = torch.cat(vid_features_part_list, dim = 0)
            return vid_features


