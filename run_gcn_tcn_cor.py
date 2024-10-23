from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from tcn import TemporalConvNet
from torch_geometric.loader import DataLoader
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from sklearn.preprocessing import MinMaxScaler


class GCN_TCN_Model(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN_TCN_Model, self).__init__()

        # GCN layers with BatchNorm
        self.gcn1 = GCNConv(num_node_features, 64)
        self.bn1 = BatchNorm(64)
        self.gcn2 = GCNConv(64, 128)
        self.bn2 = BatchNorm(128)
        self.gcn3 = GCNConv(128, 128)
        self.bn3 = BatchNorm(128)

        # TCN layer for temporal features
        # Here num_inputs corresponds to the number of node features from GCN
        self.tcn = TemporalConvNet(num_inputs=28, num_channels=[128,64, 32], kernel_size=3, dropout=0.2, norm_strategy='batchnorm')


        # Linear layer to output a prediction value
        self.fc1 = torch.nn.Linear(32, 128)
        self.fc = torch.nn.Linear(128, 1)  # Output size is 1 since it is a regression task

    def forward(self, data):
        # Graph features (GCN)
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.bn2(self.gcn2(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.bn3(self.gcn3(x, edge_index)))

        # Reshape GCN output to feed into TCN
        # The output should have the shape: [batch_size, num_time_steps, num_channels]
        batch_size = data.num_graphs
        num_nodes = x.size(0) // batch_size
        x = x.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, num_channels]

        # TCN expects [batch_size, input_channels, sequence_length], so we need to transpose
        # num_nodes就是多序列的列数，图中每个节点的num_channels就是sequence_length



        tcn_out = self.tcn(x)  # Pass through TCN layer
        tcn_out = tcn_out[:, :, -1]  # Take the output at the last time step (or take average if you want)
        tcn_out = tcn_out.view(tcn_out.size(0), -1)  # Flatten to (batch_size, feature_dim)

        # Final prediction
        out = self.fc(self.fc1(tcn_out))
        return out




df = pd.read_csv('GCN_TCN/rrzs_all.csv')


# 将最后一列作为 y，前面的列作为 x
x = df.iloc[:, 1:-1].values  # 取前面的列作为节点特征
y = df.iloc[:, -1].values   # 取最后一列作为标签
# 过滤掉 y 中的空值，并保留相应的 x
valid_indices = ~pd.isnull(y)  # 找出 y 不为空的索引
y_valid = y[valid_indices]     # 取出对应的有效 y
# 将 y_valid 进行归一化
scaler = MinMaxScaler()
y_valid = y_valid.reshape(-1, 1)  # 变为二维数组以符合 MinMaxScaler 的输入要求
y_scaled = scaler.fit_transform(y_valid)  # 对有效的 y 进行归一化


num_time_steps = 12
# 2. 定义函数：当 y 不为空时，往前取 12 行作为时间序列特征
def create_graph_data(x, y_scaled, num_time_steps):
    graph_data_list = []
    
    for i in range(num_time_steps, len(y_scaled)):
        if not pd.isnull(y_scaled[i]):  # 当 y 不为空时
            # 取前面 12 行的 x，作为节点的时间序列特征
            time_series = x[i - num_time_steps + 1:i + 1, :]  # 12 行，代表12个时间步
            node_features = torch.tensor(time_series.T, dtype=torch.float)  # 转置后，每列代表一个节点，行为时间步

            num_nodes = node_features.size(0)  # 转置后行数为节点数

            # 计算 Pearson 相关系数
            # 注意：我们需要计算每两个节点的时间序列特征之间的 Pearson 相关系数
            adj = np.corrcoef(time_series.T)  # 计算节点特征的 Pearson 相关矩阵
            adj = np.nan_to_num(adj)  # 将可能出现的 NaN 替换为 0

            # 创建稀疏图结构
            edge_index, edge_weight = dense_to_sparse(torch.tensor(adj, dtype=torch.float))  # 稀疏图结构

            # 转换为 PyTorch 张量
            label = torch.tensor([y_scaled[i]], dtype=torch.float)

            # 创建 PyG 的 Data 对象，将 node_features 作为节点特征
            graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, y=label, num_nodes=num_nodes)
            graph_data_list.append(graph_data)
    
    return graph_data_list


# 3. 创建图数据
graph_data_list = create_graph_data(x, y_scaled,num_time_steps)

# 4. 按 8:2 比例划分训练集和测试集
train_size = int(0.8 * len(graph_data_list))
train_data = graph_data_list[:train_size]
test_data = graph_data_list[train_size:]

# 5. 使用 PyG 的 DataLoader 划分 batch 大小为 32
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 打印一下训练集和测试集的大小
print(f'Train dataset size: {len(train_data)}')
print(f'Test dataset size: {len(test_data)}')

# 初始化模型、损失函数、优化器和学习率调度器
model = GCN_TCN_Model(num_node_features=num_time_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = torch.nn.MSELoss()  # 回归问题使用均方误差损失

# 训练函数
def train():
    model.train()
    total_loss = 0
    for _, data in enumerate(train_loader):
        # 表示896个节点中每个节点所属的图
        # print(data.batch.shape)
        x111 = len(data)
        # x 的形状会变为 [batch_size * num_nodes, num_features]
        x222 = data.x
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item() * data.num_graphs #data.num_graphs是当前batch中的图数量，这里就是batchsize
    return total_loss / len(train_loader.dataset)

# 测试函数
def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data).squeeze()
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def plot_predictions(y_true, y_pred, title=''):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.title('')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('./person1.png')
    plt.show()

def test_final(loader):
    model.eval()
    total_loss = 0
    true_values = []
    predicted_values = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data).squeeze()
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs
            # 保存真实值和预测值
            true_values.append(data.y.cpu().numpy())
            predicted_values.append(output.cpu().numpy())
        
        # 反归一化
        true_values = scaler.inverse_transform(np.concatenate(true_values, axis=0).reshape(-1, 1)).flatten()
        predicted_values = scaler.inverse_transform(np.concatenate(predicted_values, axis=0).reshape(-1, 1)).flatten()
        
        plot_predictions(true_values, predicted_values, title='')



# 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(1, 51):
    train_loss = train()
    test_loss = test(test_loader)  # 使用测试集
    scheduler.step()  # 调整学习率
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

test_final(test_loader)