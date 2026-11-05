import torch
TP=2
H=5120
F=13824
V=32000

def tp_rank():
    return 0
def pp_rank():
    return 0
def all_reduce(x,g):
    _=g
    return x
def send(x):
    _=x
def pp_recv(shape,dev):
    return torch.zeros(shape,device=dev)
def init(*shape,dev):
    w=torch.randn(*shape,device=dev)*0.02
    return w.requires_grad_()
def block(x,g,q,o,u,d):
    a=all_reduce((x@q)@o,g)
    m=all_reduce((torch.sigmoid(x@u)*(x@u))@d,g)
    return a+m
def run_step():
    dev=torch.device('cuda')
    g=object()
    q=init(H,H//TP,dev=dev)
    o=init(H//TP,H,dev=dev)
    u=init(H,F//TP,dev=dev)
    d=init(F//TP,H,dev=dev)
    emb=init(V,H,dev=dev)
    ids=torch.randint(V,(1,8),device=dev)
    x=emb[ids].mean(1)
    x=pp_recv(x.shape,dev) if pp_rank() else x
    y=block(x,g,q,o,u,d)
    if pp_rank():
        send(y)
    loss=y.mean()
    loss.backward()
    with torch.no_grad():
        for w in (q,o,u,d,emb):
            w-=1e-4*w.grad
            w.grad.zero_()

if __name__=='__main__':
    run_step()
