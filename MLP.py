import os
import torch
import random
import numpy as np
import torch.nn as nn
from tools.common_tools import set_seed

#MLP梯度爆炸
set_seed(1)  # 设置随机种子
class MLP(nn.Module):
    def __init__(self,neural_num,layers):
        super(MLP,self).__init__()
        self.linears=nn.ModuleList([nn.Linear(neural_num,neural_num,bias=True)for i in range(layers)])
        self.neural_num=neural_num

    def forward(self,x):
        for (i,linear) in enumerate(self.linears):
            x=linear(x)
        return x
    def initualize(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data)
flag=1
if flag:
    layer_nums=100
    neural_nums=256
    batch_size=16
    net=MLP(neural_nums,layer_nums)
    net.initualize()
    inputs=torch.randn((batch_size,neural_nums))
    outputs=net(input)
    print(outputs)