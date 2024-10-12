import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

if os.name == 'posix':
    device = torch.device('mps')
    print(torch.backends.mps.is_available())
    plt.rcParams['font.family'] = 'AppleGothic'
elif os.name == 'nt':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    plt.rcParams['font.family'] = 'Malgun Gothic'

plt.rcParams['axes.unicode_minus'] = 'False'

path_real_wages = 'real_wages.csv'
path_GDP = 'GDP.csv'
data1 = pd.read_csv(path_real_wages)
data2 = pd.read_csv(path_GDP)

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.l1 = nn.Linear(n_input, n_output)

        nn.init.constant_(self.l1.weight, 1.0)
        nn.init.constant_(self.l1.bias, 1.0)

    def forward(self, x):
        x1 = self.l1(x)
        return x1
    
df = [data1.iloc[index] for index in range(2)]
df.append(data2.iloc[0])

df = pd.DataFrame(df).T[1:]
df.columns = ['실질임금_증가율', '노동생산성지수_증가율', '경제성장률']

df = df.apply(pd.to_numeric, errors='coerce')
numpy_df = df.to_numpy()
tensor_df = torch.tensor(numpy_df).float()

X = tensor_df[:,:2]
Y = tensor_df[:,2]
Y = Y.view(-1,1)
print(X)
print(Y)

n_input = 2
n_output = 1
net = Net(n_input, n_output)

criterion = nn.MSELoss()

lr = 0.001
optimizer = optim.SGD(net.parameters(), lr=lr)
num_epochs = 5000
history = np.zeros((0,2))

for epoch in range(num_epochs):
    optimizer.zero_grad()

    outputs = net(X)

    loss = criterion(outputs, Y)

    loss.backward()

    optimizer.step()

    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

print(f'초기 손실값: {history[0,1]:.5f}')
print(f'최종 손실값: {history[-1,1]:.5f}')

print(net.l1.weight)
print(net.l1.bias)

plt.plot(history[1:,0], history[1:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.show()
