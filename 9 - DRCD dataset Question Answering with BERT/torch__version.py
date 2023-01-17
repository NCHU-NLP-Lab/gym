import torch
print(torch.__version__)
print(torch.version.cuda)

tensor = torch.rand(3,4)
print(f"Device tensor is stored on: {tensor.device}")
# Device tensor is stored on: cpu

print(torch.cuda.is_available())
#True

tensor = tensor.to('cuda')
print(f"Device tensor is stored on: {tensor.device}")