import numpy as np
import pandas as pd
import torch.fft
import warnings
import torch
import torch.nn as nn
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from Model import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import os
import torch
from datetime import datetime
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.optim as optim
import torch.nn.functional as F
# from mamba_ssm import Mamba



import logging
logging.basicConfig(filename='training_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 读取Excel文件
# df = pd.read_excel('combined_data.xlsx') # combined_data 为mixup后的数据7200条，wavelet_all为原来数据3600条
# df = pd.read_excel('wavelet_all.xlsx') # wavelet_all为原来数据3600条
# df = pd.read_excel('aaa.xlsx') # wavelet_all为原来数据3600条
# df = pd.read_excel('all_label_PCA60.xlsx') # wavelet_all为原来数据3600条
# 获取除了第一列的所有列

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. 读取数据
# ----------------------------
df = pd.read_excel('all_wave_dataset_reduced100.xlsx')  # 包含标签与特征，共3600条数据

# ----------------------------
# 2. 删除全为 NaN 的行或列
# ----------------------------
df = df.dropna(axis=0, how='all')  # 删除全为 NaN 的行
df = df.dropna(axis=1, how='all')  # 删除全为 NaN 的列

# ----------------------------
# 3. 取特征与标签
# ----------------------------
columns = df.columns[1:]  # 特征列（排除标签）
features = df[columns].values
labels = df.iloc[:, 0].values  # 第一列为标签

# ----------------------------
# 4. 处理缺失值（NaN）为均值或0（更稳妥）
# ----------------------------
features = np.nan_to_num(features, nan=0.0)

# ----------------------------
# 5. 标准化处理
# ----------------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ----------------------------
# 6. 转为三维张量 [样本数, 特征数, 1]
# ----------------------------
features_3d = np.reshape(features_scaled, (features_scaled.shape[0], features_scaled.shape[1], 1))
print(f"数据维度：{features_3d.shape}")  # e.g., (3600, 180, 1)

# ----------------------------
# 7. 转为 Torch 张量并划分数据集
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
X = torch.tensor(features_3d, dtype=torch.float32).to(device)
Y = torch.tensor(labels, dtype=torch.long).to(device)

train_data, test_data, train_labels, test_labels = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f"Train: {train_data.shape}, Test: {test_data.shape}")


# columns = df.columns[1:]
#
# # 将每一行除了第一列的数据转换为数组，并放到同一个数组中
# result = []
# for _, row in df.iterrows():
#     result.append(row[columns].values.tolist())
#
# # 获取第一列数据并转换为数组
# column_array = df.iloc[:, 0].values
#
# # 分割特征和标签
# data1 = np.array(result)  # 特征数据 (2000, 180)
# labels1 = np.array(column_array)  # 标签数据 (2000)
# print(data1.shape)
#
# # 二维转三维以便输入模型
# # (2000,180) → (number, sequence_length, feature)
# data3 = np.reshape(data1, (data1.shape[0], data1.shape[1], 1))
# print(data3.shape)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# device = "cuda"  # 选择 GPU 设备
# X = torch.tensor(data3, dtype=torch.float32)
# X = X.to(device)
# Y = torch.from_numpy(labels1)
# Y = Y.to(device)
#
# # 划分训练集和测试集
# train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=0.1, random_state=42)

print(len(train_data))

# 创建训练集和测试集的 Dataset 和 DataLoader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)







# model = MambaWaveNet().to(device) # 1 创建模型实例
# model = GRUModel().to(device) # 1 创建模型实例
# model = CNNModel().to(device) # 2 创建模型实例
# model = LSTMModel().to(device) # 3 创建模型实例
# model = TransformerModel().to(device) # 4 创建模型实例
# model = CLModel().to(device)# 5 CNN+LSTM
model = CTModel().to(device)# 6 CNN+Transformer
# model = CLTModel().to(device)# 7 顺序执行CNN-LSTM-Transformer
# model = BBMixModel().to(device)# 8 并行拼接 (cnn-lstm)64+(Transformer)64=128
# model = BBCTModel().to(device)# 9 并行拼接 (cnn)64+(Transformer)64=128

# model = CNN_BiLSTM().to(device)# 10 cnn+biLSTM
# model = CNN_BiLSTM_SelfAttention().to(device)# 11 CNN_BiLSTM_SelfAttention
# model = BiLSTM().to(device)# 12 BiLSTM
# model = CNN_Only().to(device)# 13 CNN-Only


# model = DWFormerModel().to(device)# 15 DWFormerModel
# model = HybridCNN_DWFormer().to(device)# 16 HybridCNN_DWFormer




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, eps=1e-08, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
# 定义优化器，添加 weight_decay 参数实现 L2 正则化
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # 0.01 是 L2 正则化的系数
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)



train_losses = []
train_acc = []
test_acc = []
test_losses = []
epochs = []
precision_scores = []
recall_scores = []
f1_scores = []

far_scores = []
frr_scores = []
eer_scores = []
inference_times = []  # 记录推理时间

# 训练模型
best_accuracy = 0
best_epoch = 0
num_epochs =50
roc_auc_scores = []
#保存训练权重
if not os.path.exists('weight'):
    os.makedirs('weight')
#保存模型名称
logging.info(f"方法1 ：HybridCNN_DWFormer")

for epoch in range(num_epochs):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    # print(correct)
    train_losses.append(train_loss)
    train_acc.append(train_accuracy)


    # 在测试集上计算准确率
    model.eval()  # 将模型设置为评估模式
    corrects = 0
    totals = 0
    total_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    # 计算推理时间
    start_time = time.time()

    with torch.no_grad():
        for inputss, labelss in test_loader:
            outputss = model(inputss)

            loss = criterion(outputss, labelss)  # 计算损失
            total_loss += loss.item()  # 累加损失

            _, predicted = torch.max(outputss.data, 1)
            all_labels.extend(labelss.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputss, dim=1)[:, 1].cpu().numpy())  # 获取正类的概率
            corrects += (predicted == labelss).sum().item()
            totals += labelss.size(0)
        # 计算推理时间
    inference_time = (time.time() - start_time) / len(test_loader)
    inference_times.append(inference_time)


    test_accuracy = corrects / totals
    test_loss = total_loss / len(test_loader)  # 计算平均损失

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 计算 FAR（误识率） 和 FRR（拒识率）
    false_accepts = conf_matrix[0][1]  # 负类被误识别为正类（FP）
    false_rejects = conf_matrix[1][0]  # 正类被误识别为负类（FN）
    total_negatives = conf_matrix[0][0] + conf_matrix[0][1]  # 负类总数（TN + FP）
    total_positives = conf_matrix[1][0] + conf_matrix[1][1]  # 正类总数（FN + TP）

    FAR = false_accepts / total_negatives if total_negatives != 0 else 0
    FRR = false_rejects / total_positives if total_positives != 0 else 0
    far_scores.append(FAR)
    frr_scores.append(FRR)

    # 计算 EER（等误率）
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr
    eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
    EER = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2
    eer_scores.append(EER)

    # 计算 ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    roc_auc_scores.append(roc_auc)


    test_acc.append(test_accuracy)
    test_losses.append(test_loss)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    epochs.append(epoch + 1)

    print(f"Epoch {epoch + 1}/{num_epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, "
          f"FAR: {FAR:.4f}, FRR: {FRR:.4f}, EER: {EER:.4f}, "
          f"Inference Time: {inference_time:.4f}s, ROC-AUC: {roc_auc:.4f}")

    logging.info(f"Epoch {epoch + 1}/{num_epochs}: "
                 f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                 f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
                 f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, "
                 f"FAR: {FAR:.4f}, FRR: {FRR:.4f}, EER: {EER:.4f}, "
                 f"Inference Time: {inference_time:.4f}s, ROC-AUC: {roc_auc:.4f}")


    # print(
    #     f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f},Test Accuracy: {test_accuracy:.4f},Precision: {precision:.4f},Recall: {recall:.4f},F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # # 保存日志
    #
    # logging.info(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f},Test Accuracy: {test_accuracy:.4f},Precision: {precision:.4f},Recall: {recall:.4f},F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # if (epoch + 1) % 5 == 0:
    #     torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch + 1}.pth')


    # 保存最后一轮的数据
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy}, 'final_round_data.pth')


    # 更新最佳准确率和轮数
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch + 1


# 打印最佳准确率和轮数
print(f"Best Accuracy: {best_accuracy:.4f}%, Epoch: {best_epoch}")

# 保存训练好的模型和权重
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 获取当前时间戳
model_filename = f'weight/model_epoch_{epoch + 1}_{timestamp}.pth'
torch.save(model.state_dict(), model_filename)
print(f'Model weights saved to {model_filename}')

# # 保存训练好的模型和权重
# torch.save(model.state_dict(), 'model_weights.pth')
# torch.save(model, 'entire_model.pth')



# 保存图片
# 创建保存图片的文件夹（如果文件夹不存在）
folder_path = 'save_model_figure/方法1 ：HybridCNN_DWFormer'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


plt.figure(1)
plt.plot(epochs, train_losses,label='train_loss')
plt.plot(epochs, test_losses,label='test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss and Test Loss over Epochs')
plt.legend()
# 保存为 TIFF 格式
save_path = os.path.join(folder_path, 'train_test_loss.tiff')
plt.savefig(save_path, format='tiff')
plt.show()


plt.figure(2)
plt.plot(epochs, train_acc,label='train_acc')
plt.plot(epochs, test_acc,label='test_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('train_acc and test_acc over Epochs')
plt.legend()
# 保存为 TIFF 格式
save_path = os.path.join(folder_path, 'train_test_acc.tiff')
plt.savefig(save_path, format='tiff')
plt.show()


plt.figure(3)
plt.plot(epochs, precision_scores, label='Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision over Epochs')
plt.legend()
# 保存为 TIFF 格式
save_path = os.path.join(folder_path, 'Precision.tiff')
plt.savefig(save_path, format='tiff')
plt.show()

plt.figure(4)
plt.plot(epochs, recall_scores, label='Recall')
plt.plot(epochs, f1_scores, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Recall, and F1 Score over Epochs')
plt.legend()
# 保存为 TIFF 格式
save_path = os.path.join(folder_path, 'Recall_F1.tiff')
plt.savefig(save_path, format='tiff')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
# 保存为 TIFF 格式
save_path = os.path.join(folder_path, 'confusion_matrix.tiff')
plt.savefig(save_path, format='tiff')
plt.show()

# 可视化 ROC 曲线
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
# 保存为 TIFF 格式
save_path = os.path.join(folder_path, 'ROC curve.tiff')
plt.savefig(save_path, format='tiff')
plt.show()



# 绘制 FAR、FRR、EER 变化曲线
plt.figure()
plt.plot(epochs, far_scores, label="FAR")
plt.plot(epochs, frr_scores, label="FRR")
plt.plot(epochs, eer_scores, label="EER")
plt.xlabel("Epoch")
plt.ylabel("Rate")
plt.title("FAR, FRR, and EER over Epochs")
plt.legend()
save_path = os.path.join(folder_path, 'FAR_FRR_EER.tiff')
plt.savefig(save_path, format='tiff')
plt.show()

# 绘制推理时间
plt.figure()
plt.plot(epochs, inference_times, label="Inference Time")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.title("Inference Time over Epochs")
plt.legend()
save_path = os.path.join(folder_path, 'Inference_Time.tiff')
plt.savefig(save_path, format='tiff')
plt.show()