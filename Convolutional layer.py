import torch as t
from torch.autograd import Variable as V
from PIL import Image
from torch import nn
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
