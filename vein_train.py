import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from Model import *
import os
import logging
from datetime import datetime

# ===== 1. Data Reading =====
def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format.")

    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values
    return features, labels

# ===== 2. Dataset Definition =====
class VeinDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===== 4. Training function =====
def train_model(model, train_loader, val_loader, device, epochs=200, lr=1e-3):
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    precision_scores, recall_scores, f1_scores = [], [], []
    far_scores, frr_scores, eer_scores = [], [], []
    inference_times = []
    best_accuracy, best_epoch = 0, 0
    all_labels, all_probs = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # print(f"X_batch shape: {X_batch.shape}")
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        acc = 100. * correct / total
        train_losses.append(train_loss)
        train_acc.append(acc)

        model.eval()
        val_correct, val_total = 0, 0
        val_loss = 0
        preds, trues, probs = [], [], []
        start_time = time.time()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
                preds.extend(predicted.cpu().numpy())
                trues.extend(y_batch.cpu().numpy())
                probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        end_time = time.time()
        inference_times.append(end_time - start_time)

        val_acc = 100. * val_correct / val_total
        test_losses.append(val_loss)
        test_acc.append(val_acc)
        precision_scores.append(precision_score(trues, preds, zero_division=0))
        recall_scores.append(recall_score(trues, preds, zero_division=0))
        f1_scores.append(f1_score(trues, preds, zero_division=0))

        all_labels = trues
        all_probs = probs

        # Simplified Calculation of FAR, FRR, EER
        FAR = sum((np.array(preds) == 1) & (np.array(trues) == 0)) / (np.array(trues) == 0).sum()
        FRR = sum((np.array(preds) == 0) & (np.array(trues) == 1)) / (np.array(trues) == 1).sum()
        EER = (FAR + FRR) / 2
        far_scores.append(FAR)
        frr_scores.append(FRR)
        eer_scores.append(EER)

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch + 1


        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}% "
            f"Precision: {np.mean(precision_scores):.4f}, "
            f"Recall: {np.mean(recall_scores):.4f}, "
            f"F1: {np.mean(f1_scores):.4f}, "
            f"FAR: {np.mean(far_scores):.4f}, "
            f"FRR: {np.mean(frr_scores):.4f}, "
            f"EER: {np.mean(eer_scores):.4f}, "
            f"Inference Time: {np.mean(inference_times):.4f}s"
        )
        logging.info(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}% "
            f"Precision: {np.mean(precision_scores):.4f}, "
            f"Recall: {np.mean(recall_scores):.4f}, "
            f"F1: {np.mean(f1_scores):.4f}, "
            f"FAR: {np.mean(far_scores):.4f}, "
            f"FRR: {np.mean(frr_scores):.4f}, "
            f"EER: {np.mean(eer_scores):.4f}, "
            f"Inference Time: {np.mean(inference_times):.4f}s"
        )



    print(f"Best Accuracy: {best_accuracy:.4f}%, Epoch: {best_epoch}")

    folder_path = 'save_model_figure/Add_Mix/Expriment-3 ：Remove Step 3 (Transformer encoder) '
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    epoch_range = list(range(1, epochs + 1))

    plt.figure(1)
    plt.plot(epoch_range, train_losses, label='train_loss')
    plt.plot(epoch_range, test_losses, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Test Loss over Epochs')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'train_test_loss.tiff'), format='tiff')
    plt.show()

    plt.figure(2)
    plt.plot(epoch_range, train_acc, label='train_acc')
    plt.plot(epoch_range, test_acc, label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('train_acc and test_acc over Epochs')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'train_test_acc.tiff'), format='tiff')
    plt.show()

    plt.figure(3)
    plt.plot(epoch_range, precision_scores, label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision over Epochs')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'Precision.tiff'), format='tiff')
    plt.show()

    plt.figure(4)
    plt.plot(epoch_range, recall_scores, label='Recall')
    plt.plot(epoch_range, f1_scores, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Recall and F1 Score over Epochs')
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'Recall_F1.tiff'), format='tiff')
    plt.show()

    conf_matrix = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(folder_path, 'confusion_matrix.tiff'), format='tiff')
    plt.show()

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(folder_path, 'ROC curve.tiff'), format='tiff')
    plt.show()

    plt.figure()
    plt.plot(epoch_range, far_scores, label="FAR")
    plt.plot(epoch_range, frr_scores, label="FRR")
    plt.plot(epoch_range, eer_scores, label="EER")
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.title("FAR, FRR, and EER over Epochs")
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'FAR_FRR_EER.tiff'), format='tiff')
    plt.show()

    plt.figure()
    plt.plot(epoch_range, inference_times, label="Inference Time")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Inference Time over Epochs")
    plt.legend()
    plt.savefig(os.path.join(folder_path, 'Inference_Time.tiff'), format='tiff')
    plt.show()

# ===== 5. Main Program =====
if __name__ == "__main__":
    # file_path = "data/all_wave_dataset_reduced100.xlsx"
    file_path = "data/mix_data.xlsx"                           # mix
    # file_path = "data/all_wave_dataset_reduced100.xlsx"      # no mix
    features, labels = load_data(file_path)

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

# Add Save Log
    log_dir = 'logs/Add_Mix'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'vein_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    batch_size = 64
    train_dataset = VeinDataset(X_train, y_train)
    val_dataset = VeinDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    model = MultiScaleDWFormer().to(device)

#
#     model = Ablation_NoStep1().to(device)   #Remove Step 1 (Multi-scale CNN)
#     model = Ablation_NoStep2().to(device)   #Remove Step 2 (Dynamic Attention)
#     model = Ablation_NoStep3().to(device)   #Remove Step 3 (Transformer encoder)


    train_model(model, train_loader, val_loader, device)





