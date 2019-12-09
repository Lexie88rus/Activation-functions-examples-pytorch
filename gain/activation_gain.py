import torch
import torch.nn.functional as F
import sys

# see pytorch forum topic:
# https://discuss.pytorch.org/t/calculate-gain-tanh/20854

# 1.5212 looks like a good gain value for SiLU

def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input)

a = torch.randn(1000,1000, requires_grad=True)
b = a
print (f"in: {a.std().item():.4f}")
for i in range(100):
    l = torch.nn.Linear(1000,1000, bias=False)
    torch.nn.init.xavier_normal_(l.weight, float(sys.argv[2]))
    #b = getattr(F, sys.argv[1])(l(b))
    b = silu(l(b))

    if i % 10 == 0:
        print (f"out: {b.std().item():.4f}", end=" ")
        a.grad = None
        b.sum().backward(retain_graph=True)
        print (f"grad: {a.grad.abs().mean().item():.4f}")
