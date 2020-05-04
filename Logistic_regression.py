import torch
import torch as t
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.autograd import Variable as V
# torch.manual_seed(10)
# #create data-------------
# sample_nums=100
# mean_value=1.7
# bias=1
# n_data=torch.ones(sample_nums,2)
# x0=torch.normal(mean_value*n_data,1)+bias
# y0=torch.zeros(sample_nums)
# x1=torch.normal(-mean_value*n_data,1)+bias
# y1=torch.ones(sample_nums)
# train_x=torch.cat((x0,x1),0)
# train_y=torch.cat((y0,y1),0)
# # print(train_x,"\n",train_y)
# #choose model----------------
# class LR(nn.Module):
#     def __init__(self):
#         super(LR,self).__init__()
#         self.features=nn.Linear(2,1)
#         self.sigmod=nn.Sigmoid()
#     def forward(self,x):
#         x=self.features(x)
#         x=self.sigmod(x)
#         return x
#
# lr_net=LR()
# #MCE loss
# loss_fn=nn.BCELoss()
# #optimize
# lr=0.01
# optimizer=torch.optim.SGD(lr_net.parameters(),lr=lr,momentum=0.9)
# #train model
# for iteration in range(1000):
#     y_pred=lr_net(train_x)
#     loss=loss_fn(y_pred.squeeze(),train_y)
#     optimizer.step()
#     if iteration %20 ==0:
#         mask=y_pred.ge(0.5).float().squeeze()
#         correct=(mask==train_y).sum()
#         acc=correct.item()/train_y.size(0)
#         plt.scatter(x0.data.numpy()[:,0], x0.data.numpy()[:,1],c='r',label='class 0')
#         plt.scatter(x1.data.numpy()[:,0],x1.data.numpy()[:,1],c='b',label='class 1')
#         w0,w1=lr_net.features.weight[0]
#         w0,w1=float(w0.item()),float(w1.item())
#         plot_b=float(lr_net.features.bias[0].item())
#         plot_x=np.arange(-6,6,0.1)
#         plot_y=(-w0*plot_x-plot_b)/w1
#         plt.xlim(-5,7)
#         plt.ylim(-7,7)
#         plt.plot(plot_x,plot_y)
#         plt.text(-5,5,'Loss=%.4f'%loss.data.numpy(),fontdict={'size':20,'color':'red'})
#
#        # plt.text(2, 15, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
#         plt.title("lteration:{}\nw0:{:.2f} w1:{:.2f}b:{:.2f}accurary:{:.2f}".format(iteration, w0,w1,plot_b,acc))
#         plt.legend()
#         plt.show()
#         plt.pause(0.5)
#         if acc>0.99:
#             break

#------------------------------------------------------------------
# # lihang
# class Logistic_regression():
#     def __init__(self,max_iter=200,learning_rate=0.01):
#         self.max_iter=max_iter
#         self.learning_rate=learning_rate
#
#     def sigmoid(self,x):
#         return 1/(1+exp(-1))
#     def data_matrix(self,X):
#         data_mat=[]
#         for d in X:
#             data_mat.append([1.0,*d])
#         return data_mat
a=[1,2,3,4,5]
b=[]
for i in a:
    b.append([1,*i])
print(b)
