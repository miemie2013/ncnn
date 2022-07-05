
import torch
import cv2
import numpy as np
import ncnn_utils as ncnn_utils
from my_tests.mmdet_nets import load_ckpt
from my_tests.mmgan_styleganv2ada import normalize_2nd_moment, normalize_2nd_moment2, normalize_2nd_moment2ncnn
from my_tests.mmgan_styleganv2ada import FullyConnectedLayer, StyleGANv2ADA_SynthesisNetwork

w_dim = 512
z_dim = 512
c_dim = 0
img_resolution = 512
# img_resolution = 16
img_channels = 3
channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256
synthesis = dict(
    w_dim=w_dim,
    img_resolution=img_resolution,
    img_channels=img_channels,
    channel_base=channel_base,
    channel_max=channel_max,
    num_fp16_res=num_fp16_res,
    conv_clamp=conv_clamp,
)
synthesis_ema = StyleGANv2ADA_SynthesisNetwork(**synthesis)
synthesis_ema.eval()

ckpt = torch.load('styleganv2ada_512_afhqcat.pth', map_location="cpu")
synthesis_ema = load_ckpt(synthesis_ema, ckpt["synthesis_ema"])

torch.save(synthesis_ema.state_dict(), "31.pth")

bp = open('31_pncnn.bin', 'wb')
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
bottom_names = synthesis_ema.export_ncnn(ncnn_data, bottom_names)


# 如果1个张量作为了n(n>1)个层的输入张量，应该用Split层将它复制n份，每1层用掉1个。
bottom_names = ncnn_utils.split_input_tensor(ncnn_data, bottom_names)
pp = ncnn_data['pp']
layer_id = ncnn_data['layer_id']
tensor_id = ncnn_data['tensor_id']
pp = pp.replace('tensor_%.8d' % (0,), 'images')
pp = pp.replace(bottom_names[-1], 'output')
pp = '7767517\n%d %d\n'%(layer_id, tensor_id) + pp
with open('31_pncnn.param', 'w', encoding='utf-8') as f:
    f.write(pp)
    f.close()




dic2 = np.load('30.npz')
ws = dic2['y']
ws = torch.from_numpy(ws)
ws = ws.float()
num_ws = synthesis_ema.num_ws
ws = ws.repeat([1, num_ws, 1])
ws.requires_grad_(False)
ws = ws.cuda()
synthesis_ema = synthesis_ema.cuda()

noise_mode = 'const'
y = synthesis_ema(ws, noise_mode=noise_mode)

# img = y
# img = img.permute((0, 2, 3, 1)) * 127.5 + 128
# img = img.clamp(0, 255)
# img = img.to(torch.uint8)
# img_rgb = img.cpu().detach().numpy()[0]
# img_bgr = img_rgb[:, :, [2, 1, 0]]
# cv2.imwrite('111.jpg', img_bgr)

dic = {}
dic['ws'] = ws.cpu().detach().numpy()
dic['y'] = y.cpu().detach().numpy()

np.savez('31', **dic)
print()
