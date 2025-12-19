import os
import torch
import torch.nn as nn
import pywt
import numpy as np
import cv2

# ------------------------
# 1. SE Attention Module
# ------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.pool(x).view(B, C)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(B, C, 1, 1)
        return x * y


# ------------------------
# 2. Lightweight CNN Module
# ------------------------
class ShallowCNNWithSE(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super(ShallowCNNWithSE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x


# ------------------------
# 3. 二级小波变换
# ------------------------
def wavelet_decomposition_level2(img, wavelet='haar'):
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    return [cA2, cH2, cV2, cD2, cH1, cV1, cD1]  # 7个子带


# ------------------------
# 4. 单帧特征提取
# ------------------------
def extract_features_from_frame(frame, cnn_model, device='cpu'):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    cnn_features = cnn_model(img_tensor)

    # 二级小波分解
    wavelet_feats = wavelet_decomposition_level2(img)
    wavelet_feats = [torch.from_numpy(f).float().unsqueeze(0).to(device) for f in wavelet_feats]

    wavelet_resized = []
    for wf in wavelet_feats:
        wf_up = torch.nn.functional.interpolate(wf.unsqueeze(0), size=cnn_features.shape[2:], mode='bilinear', align_corners=False)
        wavelet_resized.append(wf_up.squeeze(0))
    wavelet_stack = torch.cat(wavelet_resized, dim=0).unsqueeze(0)  # [1, 7, H, W]

    fused = torch.cat([cnn_features, wavelet_stack], dim=1)  # [1, C+7, H, W]
    se_block = SEBlock(fused.shape[1]).to(device)
    weighted = se_block(fused)

    B, C, H, W = weighted.shape
    vector = weighted.view(B, C, H * W).permute(0, 2, 1).squeeze(0)
    feature_vector = vector.flatten().detach().cpu().numpy()
    return feature_vector


# ------------------------
# 5. 视频帧序列特征提取
# ------------------------
def process_video_frames(video_path, cnn_model, device='cpu', max_frames=30):
    cap = cv2.VideoCapture(video_path)
    features = []

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        feature_vector = extract_features_from_frame(frame, cnn_model, device)
        features.append(feature_vector)
        frame_count += 1

    cap.release()

    if len(features) == 0:
        return None
    return np.stack(features)  # [T, D]


# ------------------------
# 6. 批量处理所有视频
# ------------------------
def process_all_videos(root_dir, output_npy, device='cpu'):
    cnn_model = ShallowCNNWithSE().to(device)
    cnn_model.eval()

    all_features = []
    video_names = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_path = os.path.join(root, file)
                print(f"Processing: {video_path}")

                feats = process_video_frames(video_path, cnn_model, device)
                if feats is not None:
                    all_features.append(feats)
                    video_names.append(file)

    np.savez_compressed(output_npy, features=np.array(all_features, dtype=object), filenames=np.array(video_names))
    print(f"Saved all features to {output_npy}")


# ------------------------
# 7. 主程序入口
# ------------------------
if __name__ == '__main__':
    root_dir = r"E:\deep studying\指静脉\数据库新\test_false"
    output_file = r"E:\deep studying\指静脉\数据库新\test_false_features_wavelet2.npz"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    process_all_videos(root_dir, output_file, device=device)















# import torch
# import torch.nn as nn
# import pywt
# import numpy as np
# import cv2
#
# # ------------------------
# # 1. SE Attention Module
# # ------------------------
# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(channels, channels // reduction)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(channels // reduction, channels)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         B, C, _, _ = x.size()
#         y = self.pool(x).view(B, C)
#         y = self.relu(self.fc1(y))
#         y = self.sigmoid(self.fc2(y)).view(B, C, 1, 1)
#         return x * y
#
#
# # ------------------------
# # 2. Lightweight CNN Module
# # ------------------------
# class ShallowCNNWithSE(nn.Module):
#     def __init__(self, in_channels=1, out_channels=32):
#         super(ShallowCNNWithSE, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.se = SEBlock(out_channels)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.se(x)
#         return x
#
#
# # ------------------------
# # 3. Wavelet Decomposition Function
# # ------------------------
# def wavelet_decomposition(img, wavelet='haar', level=1):
#     coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
#     cA, (cH, cV, cD) = coeffs
#     return [cA, cH, cV, cD]
#
#
# # ------------------------
# # 4. Feature Extraction Function
# # ------------------------
# def extract_features_from_frame_with_attention(frame, cnn_model, device='cpu'):
#     # Convert image to grayscale and normalize
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (128, 128))
#     img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
#     img_tensor = img_tensor.to(device)
#
#     # Extract CNN features
#     cnn_features = cnn_model(img_tensor)
#
#     # Extract wavelet subbands
#     wavelet_features = wavelet_decomposition(img, wavelet='haar', level=1)
#     wavelet_features = [torch.from_numpy(f).float().unsqueeze(0).to(device) for f in wavelet_features]
#
#     # Resize each wavelet feature to match CNN output size
#     wavelet_resized = []
#     for wf in wavelet_features:
#         wf_up = torch.nn.functional.interpolate(wf.unsqueeze(0), size=cnn_features.shape[2:], mode='bilinear', align_corners=False)
#         wavelet_resized.append(wf_up.squeeze(0))
#     wavelet_stack = torch.cat(wavelet_resized, dim=0).unsqueeze(0)  # shape: [1, 4, H, W]
#
#     # Concatenate CNN and wavelet features
#     fused = torch.cat([cnn_features, wavelet_stack], dim=1)  # shape: [1, C+4, H, W]
#
#     # Dynamic weighting via SE block
#     se_block = SEBlock(fused.shape[1]).to(device)
#     weighted = se_block(fused)
#
#     # Flatten to feature vector
#     B, C, H, W = weighted.shape
#     vector = weighted.view(B, C, H * W).permute(0, 2, 1).squeeze(0)
#     feature_vector = vector.flatten().detach().cpu().numpy()
#     return feature_vector
#
#
# # ------------------------
# # 5. Example Usage
# # ------------------------
# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     cnn_model = ShallowCNNWithSE().to(device)
#     cnn_model.eval()
#
#     # Load example image
#     image_path = 'vein_image.bmp'
#     frame = cv2.imread(image_path)
#
#     features = extract_features_from_frame_with_attention(frame, cnn_model, device=device)
#     print("Feature vector shape:", features.shape)
