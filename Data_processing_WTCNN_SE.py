import torch
import torch.nn as nn
import pywt
import numpy as np
import cv2
import os

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
# 3. 二级小波分解函数
# ------------------------
def wavelet_decomposition(img, wavelet='haar', level=2):
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    # coeffs: [cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)]
    all_bands = []
    for subband in coeffs[1:]:  # 仅取细节系数
        all_bands.extend(subband)  # cH, cV, cD
    return all_bands + [coeffs[0]]  # 顺序：[cH2, cV2, cD2, cH1, cV1, cD1, cA2]

# ------------------------
# 4. 特征提取函数
# ------------------------
def extract_features_from_frame_with_attention(frame, cnn_model, device='cpu'):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    cnn_features = cnn_model(img_tensor)

    wavelet_features = wavelet_decomposition(img, wavelet='haar', level=2)
    wavelet_features = [torch.from_numpy(f).float().unsqueeze(0).to(device) for f in wavelet_features]

    wavelet_resized = []
    for wf in wavelet_features:
        wf_up = torch.nn.functional.interpolate(wf.unsqueeze(0), size=cnn_features.shape[2:], mode='bilinear', align_corners=False)
        wavelet_resized.append(wf_up.squeeze(0))
    wavelet_stack = torch.cat(wavelet_resized, dim=0).unsqueeze(0)  # shape: [1, N, H, W]

    fused = torch.cat([cnn_features, wavelet_stack], dim=1)

    se_block = SEBlock(fused.shape[1]).to(device)
    weighted = se_block(fused)

    B, C, H, W = weighted.shape
    vector = weighted.view(B, C, H * W).permute(0, 2, 1).squeeze(0)
    feature_vector = vector.flatten().detach().cpu().numpy()
    return feature_vector

# ------------------------
# 5. Process Video File
# ------------------------
def process_video_directly(video_path, cnn_model, device='cpu'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return None

    frame_features = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            features = extract_features_from_frame_with_attention(frame, cnn_model, device)
            frame_features.append(features)
            frame_count += 1
        except Exception as e:
            print(f"Error processing frame: {e}")

    cap.release()

    if not frame_features:
        return None

    return np.mean(frame_features, axis=0)  # average over all frames

# ------------------------
# 6. Batch Process All Videos
# ------------------------
def process_all_videos(root_dir, output_npy, device='cpu'):
    cnn_model = ShallowCNNWithSE(in_channels=1, out_channels=32).to(device)
    cnn_model.eval()

    all_features = []
    video_names = []

    for subdir1 in os.listdir(root_dir):
        subdir1_path = os.path.join(root_dir, subdir1)
        if not os.path.isdir(subdir1_path):
            continue

        for subdir2 in os.listdir(subdir1_path):
            subdir2_path = os.path.join(subdir1_path, subdir2)
            if not os.path.isdir(subdir2_path):
                continue

            for video_name in os.listdir(subdir2_path):
                if video_name.endswith(('.mp4', '.avi')):
                    video_path = os.path.join(subdir2_path, video_name)
                    print(f"Processing video: {video_path}")

                    features = process_video_directly(video_path, cnn_model, device=device)
                    if features is not None:
                        all_features.append(features)
                        video_names.append(video_name)

    if not all_features:
        print("No features extracted.")
        return

    np.savez_compressed(output_npy, features=np.array(all_features), filenames=np.array(video_names))
    print(f"\nSaved all features into {output_npy}")

# ------------------------
# 7. Main Execution
# ------------------------
if __name__ == '__main__':
    root_dir = r"E:\\deep studying\\指静脉\\数据库新\\false"
    output_npy = r"E:\\deep studying\\指静脉\\数据库新\\wzd50_50\\false50_WTCNN_se.npz"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    process_all_videos(root_dir, output_npy, device=device)
