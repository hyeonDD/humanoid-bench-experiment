import torch

fp = '/aiffel/aiffel/humanoid-bench-custom-task/humanoid-bench/tdmpc2/dataset'

td = torch.load(fp)

print(td.shape)