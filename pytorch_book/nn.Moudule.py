import torch as t
from torch import nn
from torch.autograd import Variable as V
#------------全连接层
# class Linear(nn.Module):
#     def __init__(self,in_features,out_features):
#         super(Linear, self).__init__()
#         self.w=nn.Parameter(t.randn(in_features,out_features))
#         self.b=nn.Parameter(t.randn(out_features))
#
#     def forward(self,x):
#         x=x.mm(self.w)
#         return x+self.b.expand_as(x)
# layer=Linear(4,3)
# input=V(t.randn(2,4))
# #print(input)
# output=layer(input)
# #print(output)
#
# for name, parameter in layer.named_parameters():
#     print(name,parameter)
#
# class Perceptron(nn.Module):
#     def __init__(self,in_features, hidden_features, out_features):
#         nn.Module.__init__(self)
#         self.layer1=Linear(in_features, hidden_features)
#         self.layer2=Linear(hidden_features, out_features)
#     def forward(self,x):
#         x=self.layer1(x)
#         x=t.sigmoid(x)
#         return self.layer2
#
# perceptron=Perceptron(3,4,1)
# for name, param in perceptron.named_parameters():
#     print(name, param.size())
#

#---------------------conv network

from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
to_tensor=ToTensor()#IMG->TENSOR
to_pil=ToPILImage()
lena=Image.open('')
print(lena)
#input batch,batch_size=1
input=to_tensor(lena).unsqueeze(0)
#harp kernel
kernel= t.ones(3,9)/-9
kernel[1][1]=1
conv=nn.Conv2d(1,1,(3,3),1,bias=False)
conv.weight.data=kernel.view(1,1,3,3)

out=conv(V(input))
to_pil(out.data.squeeze(0))
