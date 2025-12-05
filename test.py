


''' act_scales
import torch
tensor = torch.load('act_scales/opt-125m.pt')
# print(type(tensor))
file_handle = open('log/act_opt-125m.txt', 'w')
for k, v in tensor.items():
    print(f'{k[21:]:<25} {v.shape}', file=file_handle)
file_handle.close()
'''
import torch

tensor = torch.arange(10).reshape(2, 5)
print(torch.max(tensor, dim=0))