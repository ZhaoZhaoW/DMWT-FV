conda activate vein-torch
nvcc --version  # 应该显示CUDA版本
python -c "import torch; print(torch.cuda.is_available())"  # 应该输出True
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia