
import numpy as np


dic2 = np.load('33xx.npz')
x = dic2['x']
w = dic2['w']
y = dic2['y']


import struct
bp = open('33_input.bin', 'wb')
s = struct.pack('i', 0)
bp.write(s)


out_C, in_C, kH, kW = x.shape
for i1 in range(out_C):
    for i2 in range(in_C):
        for i3 in range(kH):
            for i4 in range(kW):
                s = struct.pack('f', x[i1][i2][i3][i4])
                bp.write(s)


out_C, in_C, kH, kW = w.shape
for i1 in range(out_C):
    for i2 in range(in_C):
        for i3 in range(kH):
            for i4 in range(kW):
                s = struct.pack('f', w[i1][i2][i3][i4])
                bp.write(s)



print()

