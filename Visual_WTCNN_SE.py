import torch
import torch.nn as nn
import pywt
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    return weighted.squeeze(0), wavelet_stack.squeeze(0), img  # Return the weighted features and wavelet stack for visualization

# ------------------------
# 5. 可视化小波和CNN特征热力图
# ------------------------
def visualize_wavelet_cnn(img, cnn_map, wavelet_feats, save_dir, video_name):
    # Convert the features to numpy arrays, ensuring they are on CPU
    cnn_map = cnn_map.cpu().detach().numpy()
    wavelet_feats = wavelet_feats.cpu().detach().numpy()

    # Normalize and resize
    cnn_map = np.sum(cnn_map, axis=0)  # Summing over the channels
    cnn_map = cv2.resize(cnn_map, (img.shape[1], img.shape[0]))

    wavelet_map = np.sum(wavelet_feats, axis=0)
    wavelet_map = cv2.resize(wavelet_map, (img.shape[1], img.shape[0]))

    # Normalize for visualization
    cnn_map = np.interp(cnn_map, (cnn_map.min(), cnn_map.max()), (0, 1))
    wavelet_map = np.interp(wavelet_map, (wavelet_map.min(), wavelet_map.max()), (0, 1))

    # Create the overlay
    overlay = np.stack([wavelet_map, cnn_map, np.zeros_like(cnn_map)], axis=-1)

    # Display the results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title('Wavelet + CNN Feature Heatmap')
    plt.axis('off')

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the figure
    save_path = os.path.join(save_dir, f"{video_name}_feature_overlay.png")
    plt.savefig(save_path)
    plt.close()

# ------------------------
# 6. 处理视频文件
# ------------------------
def process_video_directly(video_path, cnn_model, device='cpu', visualize=False, vis_dir=None):
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
            cnn_map, wavelet_feats, img_resized = extract_features_from_frame_with_attention(frame, cnn_model, device)
            if visualize and vis_dir:
                visualize_wavelet_cnn(img_resized, cnn_map, wavelet_feats, vis_dir, os.path.basename(video_path))

            frame_features.append(cnn_map.flatten())
            frame_count += 1
        except Exception as e:
            print(f"Error processing frame: {e}")

    cap.release()

    if not frame_features:
        return None

    return np.mean(frame_features, axis=0)  # average over all frames

# ------------------------
# 7. 批量处理所有视频
# ------------------------
def process_all_videos(root_dir, output_npy, device='cpu', visualize=False, vis_dir=None):
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

                    features = process_video_directly(video_path, cnn_model, device=device, visualize=visualize, vis_dir=vis_dir)
                    if features is not None:
                        all_features.append(features)
                        video_names.append(video_name)

    if not all_features:
        print("No features extracted.")
        return

    np.savez_compressed(output_npy, features=all_features, video_names=video_names)
    print(f"Saved extracted features to {output_npy}")

# ------------------------
# 8. 主程序
# ------------------------
if __name__ == "__main__":
    root_dir = r"E:\\deep studying\\指静脉\\数据库新\\test_true"
    output_npy = r"E:\\deep studying\\指静脉\\数据库新\\false_features_wavelet_se.npz"
    vis_dir = r"E:\\deep studying\\指静脉\\数据库新\\feature_visualizations"  # Set the directory to save the visualizations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    process_all_videos(root_dir, output_npy, device=device, visualize=True, vis_dir=vis_dir)
