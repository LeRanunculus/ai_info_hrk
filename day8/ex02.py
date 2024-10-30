#%%
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import math
import matplotlib
import matplotlib.pyplot as plt
# %%
x = torch.linspace(0,2*math.pi,1000).unsqueeze(1)
y = torch.sin(x)
#%%
print(x)
# %%
print(y)
# %%
plt.figure(figsize=(10,5))
plt.plot(x,y,label='sin(x)',color='blue')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()
# %%
model = nn.Sequential(
        nn.Linear(1,8),
        nn.ReLU(),
        nn.Linear(8,1)
)
# %%
learing_rate = 0.01
epochs = 500
optimizer = optim.Adam(model.parameters(),lr=learing_rate)
loss_fn = nn.MSELoss()

#%%
for epoch in range(epochs) :
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"epoch {epoch} : loss {loss}")
    

# %%
import matplotlib.pyplot as plt

# test_x와 test_y를 detach()로 연산 그래프에서 분리한 후 numpy로 변환
plt.figure(figsize=(10, 5))
plt.plot(test_x, test_y.detach().numpy(), label='model(x)', color='blue')
plt.xlabel('x')
plt.ylabel('model(x)')
plt.legend()
plt.show()

# %%
