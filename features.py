import os
import cv2
import glob
import random
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn 
# from transformers import RegNetModel, RegNetConfig



# class RegNet(nn.Module):
#     def __init__(self, configuration, return_idx=[0, 1, 2, 3, 4]):
#         super(RegNet, self).__init__()  
#         self.model = RegNetModel.from_pretrained("facebook/regnet-y-040")
#         self.return_idx = return_idx


#     def forward(self, x):
        
#         outputs = self.model(x, output_hidden_states = True)
#         x = outputs.hidden_states[2:5]
#         return x

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
dataset_path = "output_segments_3"
output_features_path = "output_segments_3_resnet"
os.makedirs(output_features_path, exist_ok=True)
all_files = glob.glob(os.path.join(dataset_path, "*.mp4"))


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
feat_combine = torch.nn.Sequential(
    conv_bn_relu(64 * 3, 64, 3, padding=1, dilation=1),
    torch.nn.Conv2d(64, 64, 1),
)

feat_refine = torch.nn.Sequential(
    conv_bn_relu(64, 64, 3, padding=1, dilation=1)
)

def forward_for_squeeze(features):
    # Feature squeeze and concat
    import pdb; pdb.set_trace()
    x1 = features[0]
    x2 = features[1]
    x2 = torch.nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')
    x3 = features[2]
    x3 = torch.nn.functional.interpolate(x3, scale_factor=4, mode='bilinear')
    x_concat = torch.cat([x1, x2, x3], dim=1)
    x4 = feat_combine(x_concat)
    img_feat = feat_refine(x4)

    return img_feat
    
# Define feature extraction function
def video_to_npy(video_path, sample_rate, features_dim, model_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Select the model
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        features_dim = 2048
    # elif model_name == "regnet":
    #     model = RegNet(RegNetConfig(downsample_in_first_stage = True)).to(device)
    #     features_dim = 1088
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.eval()

    features = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:  # Process every `sample_rate`-th frame
            frame = preprocess(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model(frame)
                if model_name == "regnet":
                    feature = forward_for_squeeze(feature)

                feature = feature.view(feature.size(0), -1)
                features.append(feature)

        frame_idx += 1

    cap.release()
    # Validate feature dimensions
    for feature in features:
        if not (feature.shape == torch.Size((1, features_dim))):
            raise ValueError(f"Feature dimension mismatch: {feature.shape}, expected: (1, {features_dim})")
    
    # Combine all features into a single tensor
    features_tensor = torch.cat(features, dim=0)
    features_np = features_tensor.cpu().numpy()

    # Flatten features for saving
    flattened_frames = features_np.T
    if flattened_frames.shape[0] != features_dim:
        raise ValueError(f"Expected first dimension to be {features_dim}, but got {flattened_frames.shape[0]}")
    
    return flattened_frames

# Define data reading function
def read_data(vid_list_file, features_path, sample_rate, feature_dim, model_name):
    features_dict = {}
    random.shuffle(vid_list_file)

    for vid in tqdm(vid_list_file, desc="Extracting Features"):
        video_file = os.path.join(features_path, vid.split('.')[0] + '.mp4')
        output_file = os.path.join(output_features_path, f"{vid.split('.')[0]}.npy")

        
        features = video_to_npy(video_file, sample_rate, feature_dim, model_name)
        # import pdb; pdb.set_trace()
        np.save(output_file, features)

        features_dict[vid] = features

    return features_dict

# Main execution
if __name__ == "__main__":
    sample_rate = 1  # Process every 5th frame
    model_name = "resnet50"  # Choose between "resnet50" or "regnet"

    # Determine feature dimensions based on model
    if model_name == "resnet50":
        features_dim = 2048
    elif model_name == "regnet":
        features_dim = 1088
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    video_list = [os.path.basename(f) for f in all_files]

    # Extract features and save
    features = read_data(video_list, dataset_path, sample_rate, features_dim, model_name)
    print(f"Feature extraction complete. Total videos processed: {len(features)}")
