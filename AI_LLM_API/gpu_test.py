import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.rand(1000, 1000, device="cuda")
y = torch.rand(1000, 1000, device="cuda")

z = torch.matmul(x, y)

print("Device:", z.device)
print("VS Code AI environment working!")