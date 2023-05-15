import torch

a = torch.tensor([[[1,0,1],[0,0,1]], [[1,0,1],[0,1,1]]])
b = torch.tensor([[[1,5,9]], [[98,36,65]]])

y = a*b

print(y)
