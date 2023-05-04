import torch
import torch.nn as nn
import torch.optim as optim


# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(12, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


model = MyModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    # 生成随机数据
    x_train = torch.randn(100, 12)
    y_train = torch.randn(100, 1)

    # 前向传播和计算损失
    y_pred = model(x_train)
    # y_pred = MyModel(x_train)
    loss = criterion(y_pred, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
