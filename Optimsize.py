import torch
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable as V
#class Linear(nn.Module):
#     def __init__(self,in_features,out_features):
#         super(Linear,self).__init__()
#         self.w=nn.Parameter(t.randn(in_features,out_features))
#         self.b=nn.Parameter(t.randn(out_features))
#     def forward(self,x):
#         x=x.mm(self.w)
#         return x+self.b.expand_as(x)
# layer=Linear(4,3)
# input=V(t.randn(2,4))
# output=layer(input)
# print(output)

# arr=np.array([[1,2,3],[4,5,6]])
# t=torch.from_numpy(arr)
# print('t=',t,'\n','arr=',arr,'\n',id(t),'\n',id(arr))
# arr[0,0]=99
# print('t=',t,'\n','arr=',arr,'\n',id(t),'\n',id(arr))
torch.manual_seed(10)
lr=0.01
best_loss=float("inf")
#train data
x=torch.randn(200,1)*10
y=3*x+(5+torch.randn(200,1))
#regression parameters
w=torch.randn((1), requires_grad=True)
b=torch.zeros((1),requires_grad=True)

for iteration in range(10000):
    #forward spread
    wx=torch.mul(w,x)
    y_pred=torch.add(wx,b)
    #MSE loss
    loss=(0.5*(y-y_pred)**2).mean()
    #back spread
    loss.backward()
    # current_loss=loss.item()
    # if current_loss<best_loss:
    #         best_loss=current_loss
    #         best_w=w
    #         best_b=b
    #painting
    if loss.data.numpy()<3:
            plt.scatter(x.data.numpy(),y.data.numpy())
            plt.plot(x.data.numpy(),y_pred.data.numpy(),'r-',lw=5)
            plt.text(2,15,'Loss=%.4f'%loss.data.numpy(),fontdict={'size':20,'color':'red'})
            plt.xlim(1.5,10)
            plt.ylim(8,40)
            plt.title("lteration:{}\nw:{}b:{}".format(iteration,w.data.numpy(),b.data.numpy()))
            plt.pause(0.5)
            if loss.data.numpy()<0.55:
                break
    #renew parameters
    b.data.sub_(lr*b.grad)
    w.data.sub_(lr*w.grad)