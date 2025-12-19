import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


# 1. 从 .npz 文件中读取数据
def load_npy_data(npz_file):
    """
    从指定的 .npz 文件中加载特征矩阵和对应的视频文件名。
    """
    data = np.load(npz_file)
    features = data['features']  # 获取特征矩阵
    filenames = data['filenames']  # 获取视频文件名数组
    return features, filenames


# 2. 特征标准化处理
def normalize_features(features, method='standard'):
    """
    对特征矩阵进行标准化处理。

    参数:
        features (ndarray): 输入特征矩阵。
        method (str): 标准化方法，'standard'（默认，均值0方差1标准化）或'minmax'（归一化到[0,1]区间）。

    返回:
        normalized_features (ndarray): 标准化后的特征矩阵。
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported normalization method. Use 'standard' or 'minmax'.")

    normalized_features = scaler.fit_transform(features)
    return normalized_features


# 3. 特征降维处理（使用主成分分析PCA）
def dimensionality_reduction(features, n_components=60):
    """
    应用PCA方法对标准化后的特征进行降维。
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features


# 4. 保存降维后的特征到 Excel 文件
def save_to_excel(features, filenames, output_excel):
    """
    将降维后的特征与原始文件名组合，并保存为Excel文件。
    """
    df = pd.DataFrame(features)
    df.insert(0, 'filename', filenames)  # 将文件名插入为第一列
    df.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"Saved reduced features to {output_excel}")


# 5. 复制并生成带标签的Excel文件
def save_label_version(output_excel, label_excel, label_value=0):
    """
    读取原Excel文件，将第一列（文件名）替换为统一的标签数值，保存为新的Excel文件。
    """
    df = pd.read_excel(output_excel, engine='openpyxl')
    df.iloc[:, 0] = label_value  # 将原'filename'列全部替换为标签
    df.rename(columns={df.columns[0]: 'label'}, inplace=True)  # 重命名列为'label'
    df.to_excel(label_excel, index=False, engine='openpyxl')
    print(f"Saved labeled features (label={label_value}) to {label_excel}")


# 6. 完整流程：读取 -> 标准化 -> 降维 -> 保存原版 -> 保存标签版
def process_and_save(npz_file, output_excel, label_excel, n_components=60, label_value=0,
                     normalization_method='standard'):
    """
    从 .npz 文件读取特征并进行处理，依次保存原版特征文件及带标签版特征文件。
    """
    features, filenames = load_npy_data(npz_file)
    normalized_features = normalize_features(features, method=normalization_method)  # 新增标准化处理
    reduced_features = dimensionality_reduction(normalized_features, n_components)
    save_to_excel(reduced_features, filenames, output_excel)
    save_label_version(output_excel, label_excel, label_value)


# ========== 用法示例 ==========
if __name__ == "__main__":
    # 输入的 .npz 文件路径
    npz_file = r"E:\\deep studying\\指静脉\\数据库新\\wzd50_50\\true50_WTCNN_se.npz"

    # 输出的 .xlsx 文件路径（降维后，原版）
    output_excel = r"E:\\deep studying\\指静脉\\数据库新\\wzd50_50\\true_PCA60.xlsx"

    # 输出的 .xlsx 文件路径（带标签版）
    label_excel = r"E:\\deep studying\\指静脉\\数据库新\\wzd50_50\\true_label_PCA60.xlsx"

    # 降维后保留的特征维度
    n_components = 60

    # 标签值（例如设为1，代表False样本;0 代表真样本）
    label_value = 0

    # 标准化方法（'standard'或'minmax'）
    normalization_method = 'standard'

    # 调用完整流程
    process_and_save(npz_file, output_excel, label_excel, n_components=n_components, label_value=label_value,
                     normalization_method=normalization_method)




# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import os
#
# # 1. 从 .npz 文件中读取数据
# def load_npy_data(npz_file):
#     """
#     从指定的 .npz 文件中加载特征矩阵和对应的视频文件名。
#     """
#     data = np.load(npz_file)
#     features = data['features']  # 获取特征矩阵
#     filenames = data['filenames']  # 获取视频文件名数组
#     return features, filenames
#
# # 2. 特征降维处理（使用主成分分析PCA）
# def dimensionality_reduction(features, n_components=50):
#     """
#     对特征进行标准化处理后，应用PCA方法进行降维。
#     """
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)
#
#     pca = PCA(n_components=n_components)
#     reduced_features = pca.fit_transform(scaled_features)
#
#     return reduced_features
#
# # 3. 保存降维后的特征到 Excel 文件
# def save_to_excel(features, filenames, output_excel):
#     """
#     将降维后的特征与原始文件名组合，并保存为Excel文件。
#     """
#     df = pd.DataFrame(features)
#     df.insert(0, 'filename', filenames)  # 将文件名插入为第一列
#
#     df.to_excel(output_excel, index=False, engine='openpyxl')
#     print(f"Saved reduced features to {output_excel}")
#
# # 4. 复制并生成带标签的Excel文件（统一设为标签值0）
# def save_label_version(output_excel, label_excel, label_value=1):
#     """
#     读取原Excel文件，将第一列（文件名）替换为统一的标签数值，保存为新的Excel文件。
#     """
#     df = pd.read_excel(output_excel, engine='openpyxl')
#
#     # 将第一列 'filename' 替换为统一标签数值
#     df.iloc[:, 0] = label_value
#
#     # 重命名第一列列名为 'label'（更标准化）
#     df.rename(columns={df.columns[0]: 'label'}, inplace=True)
#
#     # 保存为新的Excel文件
#     df.to_excel(label_excel, index=False, engine='openpyxl')
#     print(f"Saved labeled features (label={label_value}) to {label_excel}")
#
# # 5. 完整流程：读取 -> 降维 -> 保存原文件 -> 保存标签文件
# def process_and_save(npz_file, output_excel, label_excel, n_components=50, label_value=1):
#     """
#     从 .npz 文件读取特征并进行处理，依次保存原版特征文件及带标签版特征文件。
#     """
#     features, filenames = load_npy_data(npz_file)
#     reduced_features = dimensionality_reduction(features, n_components)
#     save_to_excel(reduced_features, filenames, output_excel)
#     save_label_version(output_excel, label_excel, label_value)
#
# # ========== 用法示例 ==========
# if __name__ == "__main__":
#     # 输入的 .npz 文件路径
#     npz_file = r"E:\deep studying\指静脉\false_features_directly.npz"
#
#     # 输出的 .xlsx 文件路径（降维后，原版）
#     output_excel = r"E:\deep studying\指静脉\false_reduced_features.xlsx"
#
#     # 输出的 .xlsx 文件路径（带标签版）
#     label_excel = r"E:\deep studying\指静脉\false_reduced_features_label.xlsx"
#
#     # 降维后保留的特征维度
#     n_components = 50
#
#     # 标签值（当前全部设为 0）
#     label_value = 1
#
#     # 调用完整流程
#     process_and_save(npz_file, output_excel, label_excel, n_components=n_components, label_value=label_value)
#
#
#
#
# # import numpy as np
# # import pandas as pd
# # from sklearn.decomposition import PCA
# # from sklearn.preprocessing import StandardScaler
# # import os
# #
# # # 1. 从 .npz 文件中读取数据
# # def load_npy_data(npz_file):
# #     data = np.load(npz_file)
# #     features = data['features']  # 获取特征
# #     filenames = data['filenames']  # 获取视频文件名
# #     return features, filenames
# #
# # # 2. 特征降维处理（使用PCA）
# # def dimensionality_reduction(features, n_components=50):
# #     # 进行标准化处理
# #     scaler = StandardScaler()
# #     scaled_features = scaler.fit_transform(features)
# #
# #     # 使用PCA进行降维
# #     pca = PCA(n_components=n_components)
# #     reduced_features = pca.fit_transform(scaled_features)
# #
# #     return reduced_features
# #
# # # 3. 保存降维后的特征到 .xlsx 文件
# # def save_to_excel(features, filenames, output_excel):
# #     # 将特征和视频文件名结合成一个DataFrame
# #     df = pd.DataFrame(features)
# #     df.insert(0, 'filename', filenames)  # 将文件名列放在第一列
# #
# #     # 保存为 .xlsx 文件
# #     df.to_excel(output_excel, index=False, engine='openpyxl')
# #     print(f"Saved reduced features to {output_excel}")
# #
# # # 4. 完整流程：从 .npz 文件读取特征 -> 降维 -> 保存到 .xlsx
# # def process_and_save(npz_file, output_excel, n_components=50):
# #     # 加载 .npz 文件中的特征和文件名
# #     features, filenames = load_npy_data(npz_file)
# #
# #     # 对特征进行降维处理
# #     reduced_features = dimensionality_reduction(features, n_components)
# #
# #     # 保存降维后的特征到 .xlsx 文件
# #     save_to_excel(reduced_features, filenames, output_excel)
# #
# # # ========== 用法示例 ==========
# # if __name__ == "__main__":
# #     # 输入的 .npz 文件路径
# #     npz_file = r"E:\deep studying\指静脉\true_features_directly.npz"
# #     # 输出的 .xlsx 文件路径
# #     output_excel = r"E:\deep studying\指静脉\reduced_features.xlsx"
# #
# #     # 降维后保留的特征维度
# #     n_components = 50
# #
# #     # 调用完整流程
# #     process_and_save(npz_file, output_excel, n_components=n_components)
