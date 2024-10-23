## 项目说明

该项目实现了一个结合GCN（Graph Convolutional Networks）和TCN（Temporal Convolutional Networks）的模型，用于时间序列图数据的回归任务。该模型先通过GCN从图结构中提取空间特征，再通过TCN处理时间序列数据，最后进行回归预测。

### 文件结构

- `run_gcn_tcn.py`：执行训练和测试过程的主程序，包含了数据加载、模型训练、评估的完整流程。
- `run_gcn_tcn_cor.py`：与`run_gcn_tcn.py`类似，但在数据处理部分使用了相关系数来生成图结构。
- `tcn.py`：取自TCN（时序卷积网络）原论文发布的模型结构。

------

### 依赖项

请确保安装了以下依赖库：

- Python 3.x
- torch（PyTorch）
- torch_geometric
- numpy
- matplotlib
- pandas
- sklearn

可以使用以下命令安装依赖：

```
bash

pip install torch torch-geometric numpy matplotlib pandas scikit-learn
```

------

### 代码说明

#### `GCN_TCN_Model`

该模型类定义在两个主程序文件中，模型由GCN层和TCN层组成：

1. **GCN部分**：三层GCN卷积，每层后接BatchNorm层，提取空间特征。
2. **TCN部分**：将GCN输出转换为TCN输入，TCN处理多节点的时间序列信息。
3. **回归预测**：使用线性层将TCN输出转换为最终的预测值。

#### `run_gcn_tcn.py`

1. **数据加载**：从`rrzs_all.csv`文件中加载数据，使用12个时间步的数据作为节点特征。
2. **图数据生成**：通过`create_graph_data`函数，将时间序列数据转换为图结构，每个图节点的特征对应于某一时间步的特征。
3. **训练与测试**：通过`train`和`test`函数进行模型训练与测试，并计算均方误差（MSE）。
4. **评估**：在`test_final`中计算MAPE、RMSE和MAE，并绘制真实值与预测值的对比图。

#### `run_gcn_tcn_cor.py`

1. **不同之处**：数据处理部分基于皮尔逊相关系数来生成图的边（即相关性作为边权重）。
2. **其余部分**：训练、测试过程与`run_gcn_tcn.py`类似。

#### `tcn.py`

该文件定义了TCN结构，使用膨胀卷积（dilated convolution）来捕捉长时间的依赖关系，并包含批归一化（BatchNorm）或权重归一化（WeightNorm）的策略。`TemporalBlock`类实现了单层时序卷积块，`TemporalConvNet`类实现多层TCN结构。

------

### 使用指南

1. **准备数据**：将数据保存为CSV文件（如`rrzs_all.csv`），并确保最后一列为标签列。

2. 运行训练

   ：

   - 运行`run_gcn_tcn.py`或`run_gcn_tcn_cor.py`以开始训练。程序会输出训练集和测试集的损失，并在最后生成预测与真实值的对比图。
   - 训练时默认使用GPU（若可用）。

```
bash


python run_gcn_tcn.py
```

------

### 结果评估

模型会计算如下评价指标：

- **MAPE**：平均绝对百分误差
- **RMSE**：均方根误差
- **MAE**：平均绝对误差

这些指标用于评估模型的回归性能。
