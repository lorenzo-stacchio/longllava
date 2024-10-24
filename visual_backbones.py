import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torch.utils.tensorboard import SummaryWriter
# from transformers import CLIPModel, CLIPProcessor
# from dinov2 import DinoV2Model  # Assuming there is an available implementation
# from feature_extractor import Model_type
# from datasets import load_dataset
# from data_loaders.DOTA import DOTAv1
# from utils import set_seed 
# from data_loaders.utils import collate_fn 
import tqdm 
from torchvision.ops import box_iou
# import clip 
import timm
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

class Model_type(Enum):
    DinoV2 = 0
    Resnet50 = 1
    ViT = 2
    Clip_ViT = 3
    
class FeaturesExtractor(nn.Module):
    
    def __init__(self, type_model):
            super(FeaturesExtractor, self).__init__()
            self.type_model = type_model
            self.image_size = None
            self.out_channels = None
            self.model = None
            
    
    def forward(self, inputs):
        ## base implementation to be changed
        return self.model(inputs)

        
class ViTBackbone(FeaturesExtractor):
    def __init__(self, type_model):
        super(ViTBackbone, self).__init__(type_model)
        print("type_vit", self.type_model)
        self.model = timm.create_model(self.type_model, pretrained=True, num_classes=0)
        # print(self.type_model)
        if self.type_model == "vit_base_patch32_clip_448.laion2b_ft_in12k_in1k":
            # features = features.reshape(features.shape[0], 14, 14, 768)
            self.patches = 14
            self.out_channels = 768  # For vit_base_patch16_224, the token size is 768
            self.image_size = 448
        elif self.type_model == "vit_base_patch14_dinov2":
            self.patches = 37
            self.out_channels = 768  # For vit_base_patch16_224, the token size is 768        
            self.image_size = 518
        elif self.type_model == "vit_base_patch14_reg4_dinov2":
            self.patches = 37
            self.out_channels = 768  # For vit_base_patch16_224, the token size is 768
            self.image_size = 518
        elif self.type_model == "vit_base_patch32_384":
            self.patches = 12
            self.out_channels = 768  # For vit_base_patch16_224, the token size is 76
            self.image_size = 384            
        print(self.image_size)
        
    def forward(self, x):
        # Get all the features from ViT
        features = self.model.forward_features(x)
        ## remove class token
        class_token, features_tokens = features[:, 0, :], features[:, 1:, :] 
        class_token = class_token.unsqueeze(dim=1) ## to have omogeneity in toks
        # features_tokens = features_tokens.reshape(features_tokens.shape[0], self.patches, self.patches, self.out_channels)        
        return class_token, features_tokens


class DinoBackbone(FeaturesExtractor):
    def __init__(self, type_dino):
        super(DinoBackbone, self).__init__(type_dino)
        # Load the ViT model from timm with pre-trained weights
        # self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, features_only=True)
        # self.type_dino = type_dino
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('facebookresearch/dinov2', self.type_model)#.to(device=device)            
        # print(self.dino)
        self.image_size = 518
        self.out_channels = 768
        # self.model = None
        
    def forward(self, x):
        # Get all the features from ViT
        # features = self.vit(x)
        # print("iMAGE SHAPE DINO", x.shape)
        # if 544 in list(x.shape):
        #     print(x.shape)
        #     exit()

        features_total = self.model.forward_features(x)
        ## possible outs: "x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm", "masks"
        # print(len(features))
        # for el in features:
        #     print(el.shape)
        features = features_total["x_norm_patchtokens"]
        # print(patch_tokens.shape)
        ## remove class token
        # features = features[:, 1:, :] 
        # reshape by patches
        # print("dino FEATURES ", features.shape)
        if self.type_model == "dinov2_vitb14":
            features = features.reshape(features.shape[0], 37, 37, 768)
        features = features.permute(0, 3, 1, 2)

        return features
    
    
def sanity_check(model,name):
    print(f"------- SANITY CHECK for {name}-------")
    random_image_tensor = torch.rand(1, 3, model.image_size, model.image_size, dtype=torch.float32)
    # print(random_image_tensor.dtype)        
    random_image_tensor = random_image_tensor.to(device)
    class_token, features_tokens = model(random_image_tensor)
    print("FEATURE SHAPE", features_tokens.shape)
    print("class_token SHAPE", class_token.shape)
    print("------- END SANITY CHECK -------")


if __name__ == "__main__":
    type_models = ["vit_base_patch32_clip_448.laion2b_ft_in12k_in1k", "vit_base_patch14_dinov2", "vit_base_patch32_384"]
    device = "cpu"
    
    for type_model in type_models:
        model = ViTBackbone(type_model).to(device)
        sanity_check(model, type_model)