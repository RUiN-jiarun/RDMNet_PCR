# 环境安装
```sh
conda create -n rdmnet python=3.8
conda activate rdmnet
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py develop
```

# 测试方法
```sh
python experiments/exec.py --src_path example_data/A1-016-2024-03-08-14-25-16_3189.npy \
                           --ref_path example_data/A1-016-2024-04-09-13-16-17_12081.npy \
                           --snapshot weights/epoch-29.pth.tar
```