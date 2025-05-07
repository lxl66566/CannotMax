import math
import os
import random
from functools import cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@cache
def get_device(prefer_gpu=True):
    """
    获取可用的PyTorch设备

    参数:
        prefer_gpu (bool): 是否优先尝试使用GPU

    返回:
        torch.device: 可用的设备
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
            return torch.device("xpu")
    return torch.device("cpu")


device = get_device()


def preprocess_data(csv_file):
    """预处理CSV文件，将异常值修正为合理范围"""
    print(f"预处理数据文件: {csv_file}")

    # 读取CSV文件
    data = pd.read_csv(csv_file, header=None, skiprows=1)
    print(f"原始数据形状: {data.shape}")

    # # 检查特征范围
    # features = data.iloc[:, :-1].astype(np.int8)
    # labels = data.iloc[:, -1].astype(np.int8)

    # # 统计极端值
    # extreme_values = (np.abs(features) > 20).sum().sum()
    # if extreme_values > 0:
    #     print(f"发现 {extreme_values} 个绝对值大于20的特征值")

    # # 检查标签
    # invalid_labels = labels.apply(lambda x: x not in ["L", "R"]).sum()
    # if invalid_labels > 0:
    #     print(f"发现 {invalid_labels} 个无效标签")

    # # 输出特征的范围信息
    # feature_min = features.min().min()
    # feature_max = features.max().max()
    # feature_mean = features.mean().mean()
    # feature_std = features.std().mean()

    # print(f"特征值范围: [{feature_min}, {feature_max}]")
    # print(f"特征值平均值: {feature_mean:.4f}, 标准差: {feature_std:.4f}")

    # # 如果需要，可以在这里对数据进行更多的预处理
    # # 例如：将极端值截断到合理范围

    return data.shape[1]


class ArknightsDataset(Dataset):
    def __init__(self, csv_file, max_value=None, data_enhance=True, augment_factor=7):
        data = pd.read_csv(csv_file, header=None, skiprows=1)
        features = data.iloc[:, :112].values.astype(np.float32)
        labels = data.iloc[:, 112].map({"L": 0, "R": 1}).values.astype(np.int8)
        labels = np.where((labels != 0) & (labels != 1), 0, labels).astype(np.float32)

        # 分割双方单位
        feature_count = features.shape[1]
        midpoint = feature_count // 2
        left_features = features[:, :midpoint]
        right_features = features[:, midpoint:]

        # 记录原始数据的长度
        self.original_length = len(labels)
        # 用于标记每个样本是否为增强数据的掩码
        self.augmented_mask = np.zeros(self.original_length, dtype=bool)

        if data_enhance:
            # 计算需要增强的数据量
            mirror_indices = np.where(
                (left_features[:, 17] == 0) & (right_features[:, 17] == 0)
            )[0]
            enhance_size = len(mirror_indices)
            print(f"镜像增强的数据量: {enhance_size}")

            # 直接扩展原数组
            left_features = np.vstack((left_features, right_features[mirror_indices]))
            right_features = np.vstack(
                (right_features, left_features[: self.original_length][mirror_indices])
            )
            labels = np.append(labels, 1 - labels[mirror_indices])

        # 扩展增强
        if data_enhance:
            original_length = len(labels)
            all_new_lefts = []
            all_new_rights = []
            all_new_labels = []

            for i in range(original_length):
                left = left_features[i]
                right = right_features[i]
                label = labels[i]

                # 找出非零维度
                non_zero_dims = (
                    np.nonzero(left)[0] if label == 0 else np.nonzero(right)[0]
                )
                per_factor = math.ceil(len(non_zero_dims) / augment_factor)

                for dim in non_zero_dims:
                    # 获取原始值并计算增强范围
                    original_val = int(left[dim] if label == 0 else right[dim])
                    min_val = original_val + 1
                    max_val = max(int(original_val * 1.5), min_val + 1)

                    # 随机采样新值
                    sample_size = min(per_factor, max_val - min_val)
                    if sample_size <= 0:
                        continue

                    new_x = random.sample(range(min_val, max_val), sample_size)

                    # 创建增强数据
                    for x in new_x:
                        if label == 0:  # L
                            new_left = left.copy()
                            new_left[dim] = x
                            all_new_lefts.append(new_left)
                            all_new_rights.append(right)
                        else:  # R
                            new_right = right.copy()
                            new_right[dim] = x
                            all_new_lefts.append(left)
                            all_new_rights.append(new_right)
                        all_new_labels.append(label)

            # 一次性添加所有增强数据
            if len(all_new_lefts) > 0:
                left_features = np.vstack((left_features, np.array(all_new_lefts)))
                right_features = np.vstack((right_features, np.array(all_new_rights)))
                labels = np.append(labels, all_new_labels)

            # 更新增强数据掩码
            self.augmented_mask = np.concatenate(
                [
                    np.zeros(self.original_length, dtype=bool),  # 原始数据
                    np.ones(len(labels) - self.original_length, dtype=bool),  # 增强数据
                ]
            )

            print(
                f"增强后数据形状: {left_features.shape}, {right_features.shape}, {labels.shape}"
            )

        # 继续原有的处理逻辑
        if max_value is not None:
            left_features = np.clip(left_features, 0, max_value)
            right_features = np.clip(right_features, 0, max_value)

        # 转换为 PyTorch 张量并加载到 GPU
        self.left_counts = torch.from_numpy(left_features).to(device)
        self.right_counts = torch.from_numpy(right_features).to(device)
        self.left_signs = torch.from_numpy(np.sign(left_features)).to(device)
        self.right_signs = torch.from_numpy(np.sign(right_features)).to(device)
        self.labels = torch.from_numpy(labels).float().to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.left_signs[idx],
            self.left_counts[idx],
            self.right_signs[idx],
            self.right_counts[idx],
            self.labels[idx],
        )


class UnitAwareTransformer(nn.Module):
    def __init__(self, num_units, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 嵌入层
        self.unit_embed = nn.Embedding(num_units, embed_dim)
        nn.init.normal_(self.unit_embed.weight, mean=0.0, std=0.02)

        self.value_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # 注意力层与FFN
        self.enemy_attentions = nn.ModuleList()
        self.friend_attentions = nn.ModuleList()
        self.enemy_ffn = nn.ModuleList()
        self.friend_ffn = nn.ModuleList()

        for _ in range(num_layers):
            # 敌方注意力层
            self.enemy_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=0.2
                )
            )
            self.enemy_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 友方注意力层
            self.friend_attentions.append(
                nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, dropout=0.2
                )
            )
            self.friend_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

            # 初始化注意力层参数
            nn.init.xavier_uniform_(self.enemy_attentions[-1].in_proj_weight)
            nn.init.xavier_uniform_(self.friend_attentions[-1].in_proj_weight)

        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1)
        )

    def forward(self, left_sign, left_count, right_sign, right_count):
        # 提取Top3兵种特征
        left_values, left_indices = torch.topk(left_count, k=3, dim=1)
        right_values, right_indices = torch.topk(right_count, k=3, dim=1)

        # 嵌入
        left_feat = self.unit_embed(left_indices)  # (B, 3, 128)
        right_feat = self.unit_embed(right_indices)  # (B, 3, 128)

        embed_dim = self.embed_dim

        # 前x维不变，后y维 *= 数量，但使用缩放后的值
        left_feat = torch.cat(
            [
                left_feat[..., : embed_dim // 2],  # 前x维
                left_feat[..., embed_dim // 2 :]
                * left_values.unsqueeze(-1),  # 后y维乘数量
            ],
            dim=-1,
        )
        right_feat = torch.cat(
            [
                right_feat[..., : embed_dim // 2],
                right_feat[..., embed_dim // 2 :] * right_values.unsqueeze(-1),
            ],
            dim=-1,
        )

        # FFN
        left_feat = left_feat + self.value_ffn(left_feat)
        right_feat = right_feat + self.value_ffn(right_feat)

        # 生成mask (B, 3) 0.1防一手可能的浮点误差
        left_mask = left_values > 0.1
        right_mask = right_values > 0.1

        for i in range(self.num_layers):
            # 敌方注意力
            delta_left, _ = self.enemy_attentions[i](
                query=left_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )
            delta_right, _ = self.enemy_attentions[i](
                query=right_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.enemy_ffn[i](left_feat)
            right_feat = right_feat + self.enemy_ffn[i](right_feat)

            # 友方注意力
            delta_left, _ = self.friend_attentions[i](
                query=left_feat,
                key=left_feat,
                value=left_feat,
                key_padding_mask=~left_mask,
                need_weights=False,
            )
            delta_right, _ = self.friend_attentions[i](
                query=right_feat,
                key=right_feat,
                value=right_feat,
                key_padding_mask=~right_mask,
                need_weights=False,
            )

            # 残差连接
            left_feat = left_feat + delta_left
            right_feat = right_feat + delta_right

            # FFN
            left_feat = left_feat + self.friend_ffn[i](left_feat)
            right_feat = right_feat + self.friend_ffn[i](right_feat)

        # 输出战斗力
        L = self.fc(left_feat).squeeze(-1) * left_mask
        R = self.fc(right_feat).squeeze(-1) * right_mask

        # 计算战斗力差输出概率，'L': 0, 'R': 1，R大于L时输出大于0.5
        output = torch.sigmoid(R.sum(1) - L.sum(1))

        return output


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for ls, lc, rs, rc, labels in train_loader:
        ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]

        optimizer.zero_grad()

        # 检查输入值范围
        if (
            torch.isnan(ls).any()
            or torch.isnan(lc).any()
            or torch.isnan(rs).any()
            or torch.isnan(rc).any()
        ):
            print("警告: 输入数据包含NaN，跳过该批次")
            continue

        if (
            torch.isinf(ls).any()
            or torch.isinf(lc).any()
            or torch.isinf(rs).any()
            or torch.isinf(rc).any()
        ):
            print("警告: 输入数据包含Inf，跳过该批次")
            continue

        # 确保labels严格在0-1之间
        if (labels < 0).any() or (labels > 1).any():
            print("警告: 标签值不在[0,1]范围内，进行修正")
            labels = torch.clamp(labels, 0, 1)

        try:
            outputs = model(ls, lc, rs, rc).squeeze()

            # 确保输出在合理范围内
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("警告: 模型输出包含NaN或Inf，跳过该批次")
                continue

            # 确保输出严格在0-1之间，因为BCELoss需要
            if (outputs < 0).any() or (outputs > 1).any():
                print("警告: 模型输出不在[0,1]范围内，进行修正")
                outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

            loss = criterion(outputs, labels)

            # 检查loss是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 损失值为 {loss.item()}, 跳过该批次")
                continue

            loss.backward()

            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        except RuntimeError as e:
            print(f"警告: 训练过程中出错 - {str(e)}")
            continue

    return total_loss / max(1, len(train_loader)), 100 * correct / max(1, total)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for ls, lc, rs, rc, labels in data_loader:
            ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]

            # 检查输入值范围
            if (
                torch.isnan(ls).any()
                or torch.isnan(lc).any()
                or torch.isnan(rs).any()
                or torch.isnan(rc).any()
                or torch.isinf(ls).any()
                or torch.isinf(lc).any()
                or torch.isinf(rs).any()
                or torch.isinf(rc).any()
            ):
                print("警告: 评估时输入数据包含NaN或Inf，跳过该批次")
                continue

            # 确保labels严格在0-1之间
            if (labels < 0).any() or (labels > 1).any():
                labels = torch.clamp(labels, 0, 1)

            try:
                outputs = model(ls, lc, rs, rc).squeeze()

                # 确保输出在合理范围内
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print("警告: 评估时模型输出包含NaN或Inf，跳过该批次")
                    continue

                # 确保输出严格在0-1之间，因为BCELoss需要
                if (outputs < 0).any() or (outputs > 1).any():
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)

                loss = criterion(outputs, labels)

                # 检查loss是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            except RuntimeError as e:
                print(f"警告: 评估过程中出错 - {str(e)}")
                continue

    return total_loss / max(1, len(data_loader)), 100 * correct / max(1, total)


def stratified_random_split(dataset: ArknightsDataset, test_size=0.1, seed=42):
    """
    将数据集分割为训练集和测试集，确保增强数据只进入训练集

    参数:
        dataset: ArknightsDataset 实例
        test_size: 测试集比例
        seed: 随机种子

    返回:
        train_subset: 训练集 Subset
        test_subset: 测试集 Subset
    """
    # 获取标签并转换为numpy数组
    labels = dataset.labels
    if labels.device != torch.device("cpu"):
        labels = labels.cpu()
    labels = labels.numpy()

    # 获取原始数据和增强数据的索引
    original_indices = np.where(~dataset.augmented_mask)[0]
    augmented_indices = np.where(dataset.augmented_mask)[0]

    # 只对原始数据进行分层抽样
    original_labels = labels[original_indices]
    train_orig_indices, test_indices = train_test_split(
        original_indices,
        test_size=test_size,
        random_state=seed,
        stratify=original_labels,
    )

    # 将所有增强数据添加到训练集
    train_indices = np.concatenate([train_orig_indices, augmented_indices])

    # 转换为列表以保持与你原始代码一致
    train_indices = train_indices.tolist()
    test_indices = test_indices.tolist()

    # 创建Subset对象
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    return train_subset, test_subset


def main():
    # 配置参数
    config = {
        # 数据文件路径
        "data_file": "66kfpdd.csv",
        # 训练时的批量大小，影响内存使用和训练速度，128不够用了
        "batch_size": 4096,
        # 测试集比例（10%的数据作为测试集）
        "test_size": 0.1,
        # 嵌入层的维度大小（特征表示的维度）128不够用了，512会过拟合
        "embed_dim": 256,
        # Transformer的层数（堆叠的编码器/解码器层数量）
        "n_layers": 6,
        # 多头注意力机制中的头数
        "num_heads": 8,
        # 学习率，控制参数更新的步长
        "lr": 2e-4,
        # 训练的总轮次
        "epochs": 70,
        # 随机种子，用于保证实验可重复性
        "seed": 1999,
        # 模型保存目录
        "save_dir": "models",
        # 特征值的最大限制，用于数据预处理，防止极端值影响模型稳定性
        "max_feature_value": 80,
    }

    # 创建保存目录
    os.makedirs(config["save_dir"], exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    # 设置设备
    print(f"使用设备: {device}")

    # 检查CUDA可用性
    if device != "cpu":
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
            print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

        # 设置确定性计算以增加稳定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("警告: 未检测到GPU，将在CPU上运行训练，这可能会很慢!")

    # 先预处理数据，检查是否有异常值
    num_data = preprocess_data(config["data_file"])

    # 加载数据集
    dataset = ArknightsDataset(
        config["data_file"],
        max_value=config["max_feature_value"],  # 使用最大值限制
    )

    # 划分
    train_dataset, val_dataset = stratified_random_split(
        dataset, test_size=config["test_size"], seed=config["seed"]
    )

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
    )

    # 初始化模型
    model = UnitAwareTransformer(
        num_units=(num_data - 1) // 2,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["n_layers"],
    ).to(device)

    print(
        f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # 训练历史记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 训练设置
    best_acc = 0
    best_loss = float("inf")

    # 训练循环
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 更新学习率
        scheduler.step()

        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存最佳模型（基于准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(config["save_dir"], "best_model_acc.pth"),
            )
            torch.save(model, os.path.join(config["save_dir"], "best_model_full.pth"))
            print("保存了新的最佳准确率模型!")

        # 保存最佳模型（基于损失）
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(config["save_dir"], "best_model_loss.pth"),
            )
            print("保存了新的最佳损失模型!")

        # 保存最新模型
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "config": config,
            },
            os.path.join(config["save_dir"], "latest_checkpoint.pth"),
        )

        # 打印训练信息
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        print("-" * 40)

        # 绘制并保存训练历史
        # if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
        #     plot_training_history(
        #         train_losses, val_losses, train_accs, val_accs,
        #         save_path=os.path.join(config['save_dir'], 'training_history.png')
        #     )

    print(f"训练完成! 最佳验证准确率: {best_acc:.2f}%, 最佳验证损失: {best_loss:.4f}")

    # 保存最终训练历史
    # plot_training_history(
    #     train_losses, val_losses, train_accs, val_accs,
    #     save_path=os.path.join(config['save_dir'], 'final_training_history.png')
    # )


if __name__ == "__main__":
    main()
