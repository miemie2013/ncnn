
import torch
import cv2
import numpy as np
import ncnn_utils as ncnn_utils
from my_tests.mmdet_nets import load_ckpt
from my_tests.mmgan_styleganv2ada import MyConv2d


in_channels = 8
# mid_channels = 1024
out_channels = 2
kernel_size = 3
padding = 0


# model = MyConv2d(in_channels, mid_channels, out_channels, kernel_size, padding)
model = MyConv2d(in_channels, out_channels, kernel_size, padding)
model.eval()


dic2 = np.load('33xx.npz')
x = dic2['x']
w = dic2['w']
y = dic2['y']


def copy(name, w, std):
    value2 = torch.Tensor(w)
    value = std[name]
    value.copy_(value2)
    std[name] = value

model_std = model.state_dict()
copy('weight', w, model_std)
copy('conv1.weight', w, model_std)
model.load_state_dict(model_std)



# torch.save(model.state_dict(), "33.pth")

bp = open('33_pncnn.bin', 'wb')
pp = ''
layer_id = 0
tensor_id = 0
pp += 'Input\tlayer_%.8d\t0 1 tensor_%.8d\n' % (layer_id, tensor_id)
layer_id += 1
tensor_id += 1

ncnn_data = {}
ncnn_data['bp'] = bp
ncnn_data['pp'] = pp
ncnn_data['layer_id'] = layer_id
ncnn_data['tensor_id'] = tensor_id
bottom_names = ncnn_utils.newest_bottom_names(ncnn_data)
bottom_names = model.export_ncnn(ncnn_data, bottom_names)


# 如果1个张量作为了n(n>1)个层的输入张量，应该用Split层将它复制n份，每1层用掉1个。
bottom_names = ncnn_utils.split_input_tensor(ncnn_data, bottom_names)
pp = ncnn_data['pp']
layer_id = ncnn_data['layer_id']
tensor_id = ncnn_data['tensor_id']
pp = pp.replace('tensor_%.8d' % (0,), 'images')
pp = pp.replace(bottom_names[-1], 'output')
pp = '7767517\n%d %d\n'%(layer_id, tensor_id) + pp
with open('33_pncnn.param', 'w', encoding='utf-8') as f:
    f.write(pp)
    f.close()



x = torch.from_numpy(x)
x = x.to(torch.float32)
x.requires_grad_(False)

x = x.cuda()
model = model.cuda()
y2 = model(x)

y2 = y2.cpu().detach().numpy()


# ddd = np.sum((y - y2) ** 2)
# print('sum  ddd=%.9f' % ddd)
#
# ddd2 = np.mean((y - y2) ** 2)
# print('mean ddd=%.9f' % ddd2)

print('x=')
print(x.cpu().detach().numpy())
print('y=')
print(y2)
dic = {}
dic['y'] = y2

np.savez('33', **dic)
print()
