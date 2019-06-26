import torch
from tqdm import tqdm
# N, D_in, D_out = 64, 1024, 512
# x = torch.randn(N, D_in, device="cuda")
# y = torch.randn(N, D_out, device="cuda")
# model = torch.nn.Linear(D_in, D_out).cuda()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# max_epoch = tqdm(range(50000))
# for t in max_epoch:
#     y_pred = model(x)
#     loss = torch.nn.functional.mse_loss(y_pred, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     max_epoch.set_postfix({'loss':loss.item()})

from apex import amp

N, D_in, D_out = 64, 1024, 512
x = torch.randn(N, D_in, device="cuda")
y = torch.randn(N, D_out, device="cuda")
model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
max_epoch = tqdm(range(50000))
for t in max_epoch:
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    optimizer.zero_grad()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    max_epoch.set_postfix({'loss': loss.item()})
