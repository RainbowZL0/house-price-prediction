"""
1. df[], df.iloc[], df.loc[]区别
2. df的行列号，和行列名是两种东西。
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset
import a000_Tools

TRAINING_PATH, TEST_PATH = "./dataset/train.csv", "./dataset/test.csv"
TENSOR_TYPE = torch.float32
LOSS_FUNC = nn.MSELoss()
TOTAL_EPOCHS, LR, BATCH_SIZE = 500, 0.1, 256
DEVICE = torch.device('cuda')
DROP_H1, DROP_H2, DROP_H3 = 0.3, 0.3, 0.3
K = 10
NUM_WORKERS = 8


def start():
    training_data, test_data = read_data()
    training_set, test_set, training_labels = prepare_data(training_data, test_data)  # 预处理
    prepare_and_train(training_set, test_set, training_labels)


def read_data():
    training_data = pd.read_csv(TRAINING_PATH)
    test_data = pd.read_csv(TEST_PATH)
    print(training_data.shape, test_data.shape)
    return training_data, test_data


def prepare_data(training_data, test_data):
    # training_data的第一列和最后一列去掉。分别是ID和label。
    # validation_data的第一列去掉，第一列是ID。
    # 然后将二者连起来成为同一个表。数字化的操作应该同时在所有数据上进行
    total_data = pd.concat((training_data.iloc[:, 1:-1],
                            test_data.iloc[:, 1:]))

    # 筛选出取值类型为数字的所有列。此处不是object的就是number的列
    # dtypes是一个序列。index为列名，值为这一列的取值类型。
    # bool_list的index是列名，和dtypes一样，因为就是从dtypes序列来的
    numerical_columns_bool_list = (total_data.dtypes != 'object')
    numerical_column_names = total_data.dtypes[numerical_columns_bool_list].index
    # 所有数字列分别做标准化z-score。分母加一个很小的数，防止分母为0
    total_data[numerical_column_names] = (
        total_data)[numerical_column_names].apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-10))
    # 然后填充NaN为均值，即刚刚标准化之后的0。上一步标准化是可以带着NaN一起做的，不能先填充为0，因为会改变原来的分布。
    total_data[numerical_column_names] = total_data[numerical_column_names].fillna(0)

    # 处理离散值 one-hot编码。然后统一整张表的数据类型，再转换为numpy，再到tensor
    total_data = pd.get_dummies(total_data, dummy_na=True).astype('float32')
    print(total_data.shape)
    num_training_data = training_data.shape[0]  # shape[0]是行数

    # total_data.to_excel('./total_data.xlsx', index=False)

    # x_data = total_data[:num_training_data]
    # x_data = x_data.astype(float)
    # df = pd.DataFrame(x_data.values)
    # df.to_excel('./training_data.xlsx', index=False)
    # print(df.dtypes.value_counts())

    # 重新划分出训练和验证。转为tensor。

    training_set = torch.tensor(total_data[:num_training_data].values,
                                dtype=TENSOR_TYPE,
                                device=DEVICE)
    test_set = torch.tensor(total_data[num_training_data:].values,
                            dtype=TENSOR_TYPE,
                            device=DEVICE)
    # loc是按名称取，如果行和列中只写一个参数会认为是筛选行，不筛选列
    training_labels = torch.tensor(training_data.loc[:, ['SalePrice']].values,
                                   dtype=TENSOR_TYPE,
                                   device=DEVICE)

    return training_set, test_set, training_labels


def prepare_and_train(training_set, test_features, training_labels):
    input_dim = training_set.shape[1]
    print(training_set.shape, training_labels.shape)

    net = MyNet(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(params=net.parameters(),
                                 lr=LR)

    current_k = 0
    for training_features, training_labels, vali_features, vali_labels in a000_Tools.k_fold_cv_iter(
            k=K,
            training_features=training_set,
            training_labels=training_labels
    ):
        # final_training_features = torch.cat((training_features, vali_features))
        # final_training_labels = torch.cat((training_labels, vali_labels))
        # training_features_labels = TensorDataset(final_training_features, final_training_labels)

        training_features_labels = TensorDataset(training_features, training_labels)
        vali_features_labels = TensorDataset(vali_features, vali_labels)
        training_dtl = DataLoader(training_features_labels,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)
        vali_dtl = DataLoader(vali_features_labels,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)
        train_and_validate(training_dtl, vali_dtl, net, optimizer, current_k)
        current_k += 1

        # break
    test_model_performance(net, test_features)


def train_and_validate(training_dtl, vali_dtl, net, optimizer, current_k):
    net.apply(weight_initialize)
    training_mse, vali_mse = 0, 0
    for epoch in range(TOTAL_EPOCHS):
        net.train()
        # 频繁append的操作最好用list，而不是tensor.cat
        loss_list = []
        y_hat_list = []
        y_list = []
        for x, y in training_dtl:
            # x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = net(x)
            loss = ratio_rmse(y_hat, y)

            loss_list.append(loss.item())
            y_hat_list.extend(y_hat)
            y_list.extend(y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 每个epoch结束之后，计算train和vali的loss
        y_hat_tensor = torch.tensor(y_hat_list,
                                    device=DEVICE,
                                    dtype=TENSOR_TYPE)
        y_tensor = torch.tensor(y_list,
                                device=DEVICE,
                                dtype=TENSOR_TYPE)
        training_mse = LOSS_FUNC(y_hat_tensor, y_tensor)

        with torch.no_grad():
            net.eval()
            vali_loss_list = []
            vali_y_hat_list = []
            vali_y_list = []
            for vali_x, vali_y in vali_dtl:
                vali_y_hat = net(vali_x)
                vali_loss = ratio_rmse(vali_y_hat, vali_y)

                vali_loss_list.append(vali_loss.item())
                vali_y_hat_list.extend(vali_y_hat.detach().cpu().tolist())
                vali_y_list.extend(vali_y.detach().cpu().tolist())
            vali_y_hat_tensor = torch.tensor(vali_y_hat_list,
                                             device=DEVICE,
                                             dtype=TENSOR_TYPE)
            vali_y_tensor = torch.tensor(vali_y_list,
                                         device=DEVICE,
                                         dtype=TENSOR_TYPE)
            vali_mse = LOSS_FUNC(vali_y_hat_tensor, vali_y_tensor)

        print(f"current_k_fold = {current_k + 1}, "
              f"epoch = {epoch + 1}, "
              f"loss = {sum(loss_list) / len(loss_list)}, "
              f"training_mse = {training_mse / 1e+8}, "
              f"vali_mse = {vali_mse / 1e+8}")

    # 训练结束，记录mse结果
    return training_mse, vali_mse


def log_rmse(y_hat, y_real):
    y_hat = torch.clamp(y_hat, 1, float('inf'))
    return torch.sqrt(LOSS_FUNC(torch.log(y_hat), torch.log(y_real)))


def ratio_rmse(y_hat, y_real):
    return torch.sqrt(LOSS_FUNC(y_hat / y_real, torch.ones(size=y_hat.shape, device=DEVICE)))


def test_model_performance(net, test_set):
    """输出test集上的预测csv文件"""
    df = pd.DataFrame()
    df.loc[:, ["Id"]] = pd.read_csv(TEST_PATH).loc[:, ["Id"]].to_numpy()
    with torch.no_grad():
        net.eval()
        y_hat = net(test_set)
    df.loc[:, ["SalePrice"]] = y_hat.detach().cpu().numpy()  # 有grad的tensor需要先detach
    df.to_csv("./answer.csv", index=False)


def weight_initialize(layer):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:  # 判断是否是线性层
        init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)


class MyNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256, dtype=TENSOR_TYPE), nn.BatchNorm1d(256), nn.LeakyReLU(),
            nn.Linear(256, 128, dtype=TENSOR_TYPE), nn.BatchNorm1d(128), nn.LeakyReLU(),
            nn.Linear(128, 64, dtype=TENSOR_TYPE), nn.BatchNorm1d(64), nn.LeakyReLU(),
            nn.Linear(64, 32, dtype=TENSOR_TYPE), nn.BatchNorm1d(32), nn.LeakyReLU(),
            nn.Linear(32, 16, dtype=TENSOR_TYPE), nn.BatchNorm1d(16), nn.LeakyReLU(),
            nn.Linear(16, 8, dtype=TENSOR_TYPE), nn.BatchNorm1d(8), nn.LeakyReLU(),
            nn.Linear(8, 1, dtype=TENSOR_TYPE)
        )

    def forward(self, x):
        return self.model(x)


def print_out_shape():
    net = MyNet(100).to(device=DEVICE)
    x = torch.zeros(size=(1, 100),
                    device=DEVICE,
                    dtype=TENSOR_TYPE)
    for every_layer in net.model:
        x = every_layer(x)
        if isinstance(every_layer, torch.nn.Linear):
            print(every_layer.weight.shape)
        print(every_layer.__class__.__name__, x.shape)


if __name__ == '__main__':
    # print_out_shape()
    start()
