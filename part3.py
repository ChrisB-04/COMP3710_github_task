import numpy as np
import math 
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Y, X = np.mgrid[-4:4:0.005, -4:4:0.005]
x = torch.Tensor(X)
y = torch.Tensor(Y)

x.to(device)
y.to(device)

def ikeda_step(x,y):
    t = 0.4 - 6 / (1 + x*x + y*y)
    x_next = 1 + 0.9 * (x * torch.cos(t) - y * torch.sin(t))
    y_next = 0.9 * (x * torch.sin(t) + y * torch.cos(t))

    return x_next, y_next

#burn in 
for _ in range(50):
    x, y = ikeda_step(x,y)

counts = torch.zeros_like(x)

threshold = 200
for _ in range(threshold):
    x, y = ikeda_step(x, y)
    
    # map coords to pixel indices
    i = ((y + 4) / 0.005).long()
    j = ((x + 4) / 0.005).long()
    
    # mask points that are outside the bounds
    mask = (i >= 0) & (i < counts.shape[0]) & (j >= 0) & (j < counts.shape[1])
    i = i[mask]
    j = j[mask]
    
    # increment counts
    counts.index_put_((i, j), torch.ones_like(i, dtype=counts.dtype), accumulate=True)


# #plot 
# fig = plt.figure(figsize=(16,10))

def processFractal(a):
    """
    Display an array of iteration counts as a colorful 
    picture of a fractal. 
    """

    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a


plt.imshow(processFractal(counts.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()


