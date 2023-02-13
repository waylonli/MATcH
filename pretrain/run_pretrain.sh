python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch._C._cuda_getArchFlags())"
python -c "import torch; print(torch.cuda.device_count())"


python train_from_scratch.py
