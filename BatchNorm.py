import torch as t
from torch import nn
from torch.autograd import Variable as V

#Linear全连接层
#BatchNorm 批规范化层
input=V(t.randn(2,3))
linear=nn.Linear(3,4)
h=linear(input)
#print(h)

bn=nn.BatchNorm1d(4)
bn.weight.data=t.ones(4)*4
bn.bias.data=t.zeros(4)
bn_out=bn(h)
bn_out.mean(0),bn_out.var(0,unbiased=False)
print(bn)