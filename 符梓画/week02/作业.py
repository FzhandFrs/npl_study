import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 超参数
input_dim = 5  # 输入向量的维度
output_dim = 5  # 输出类别数（五个类别）
batch_size = 64
epochs = 10

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
model = SimpleNN(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式

    # 生成批次数据
    inputs = torch.randn(batch_size, input_dim)  # 随机生成五维输入向量
    labels = torch.argmax(inputs, dim=1)  # 根据最大值所在的维度生成标签

    # 前向传播
    outputs = model(inputs)  # 模型输出
    loss = criterion(outputs, labels)  # 计算损失

    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 输出每个 epoch 的训练信息
    if (epoch + 1) % 2 == 0:  # 每隔2个epoch输出一次
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    test_input = torch.randn(1, input_dim)  # 生成一个测试输入
    test_output = model(test_input)
    predicted_class = torch.argmax(test_output, dim=1)  # 预测类别
    print(f'Predicted class: {predicted_class.item()}')
