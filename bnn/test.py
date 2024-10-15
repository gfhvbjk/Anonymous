import torch
b = torch.ceil(torch.log2(torch.abs(torch.tensor(0.001))))
print(b)
a = 2**b
print(a)
print(2^(-2))