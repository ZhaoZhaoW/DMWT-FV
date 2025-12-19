
# Step 1: Data Processing
wavelet_patch.py: Processing positive and negative sample data ()
wave_video_dataset100.npz: Output from wavelet_patch.py processing, featuring 60,000 dimensions

Dimension: Feature vector shape: (638976,)



# Step 2: Data Compression
wavelet_patch_excel.py: Dimension reduction
wave_dataset_reduced100.xlsx: PCA reduced to 50 dimensions

# Step 3: Data and Label Partitioning
all_wave_dataset_reduced100.xlsx: First column represents labels, followed by 50-dimensional feature data



#Step 4: Mixup Augmentation
  Expand data from `data/all_wave_dataset_reduced100.xlsx` using Mixup to `data/mix_data.xlsx`
Generate final dataset: `mix_data.xlsx`


#Step 4: Training and Visualization
vein_train.py: Main program for model training; (trains mix_data.xlsx)


#Step 5: Various Models
Model.py: Summary of various models

Translated with DeepL.com (free version)





