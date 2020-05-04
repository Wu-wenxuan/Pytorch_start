import numpy
import torch
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable as V
flag=True
if flag:
    w=torch.tensor([1.],requires_grad=True)
    x=torch.tensor([2.],requires_grad=True)
    # a=torch.add(w,x)
    # b=torch.add(w,1)
    # y0 = torch.mul(a, b)  # y0 = (x+w) * (w+1)
    # y1 = torch.add(a, b)  # y1 = (x+w) + (w+1)    dy1/dw = 2
    # loss=torch.cat([y0,y1],dim=0)
    # grad_tensors=torch.tensor([1.,2.])
    # loss.backward(gradient=grad_tensors)
    # print(w.grad)




# for i in range(4):
#     a=torch.add(w,x)
#     b=torch.add(w,1)
#     y=torch.mul(a,b)

#     y.backward()
#     #print(w.grad)
#     w.grad.zero_()
#
# a=np.arange(32).reshape((8,4))
# print(a[[-4,-2,-1,-7]])

# mean=0.7
# std=0.1
# n_data=torch.ones(5,3)
# array=np.array([[1,2],[3,4]])
# print(array)
# z=torch.normal(mean*n_data,2)+std
# print(z)