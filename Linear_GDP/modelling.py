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
    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3
    
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

n_input = 1
n_output = 1
net = Net()

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

print("l1 weight :", net.l1.weight)
print("l1 bias :",net.l1.bias)
print("l2 weight :",net.l2.weight)
print("l2 bias :",net.l2.bias)
print("l3 weight :",net.l3.weight)
print("l3 bias :",net.l3.bias)

labels_pred = net(X)

# 3D 그래프 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 3D 산점도 그리기
ax.scatter(X[:,0], X[:,1], labels_pred[:,0].data, c='b', marker='o')

# 축 이름 설정
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
"""
plt.title('은닉층 2개, 활성화 함수 사용')
plt.scatter(X.data, labels_pred[:,0].data, c='b', label='예측값')
plt.scatter(X.data, Y.data, c='k', marker='x',label='정답')
plt.legend()
plt.show()
"""
"""
plt.plot(history[1:,0], history[1:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선(손실)')
plt.show()
"""