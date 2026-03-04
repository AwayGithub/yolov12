import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA编译版本: {torch.version.cuda}")
print(f"CUDA运行时是否可用: {torch.cuda.is_available()}")