import os

import torch
import torch.serialization
from torch.utils.data import DataLoader

from train import ArknightsDataset, UnitAwareTransformer, get_device


def test_model(model_path, data_file, batch_size=1024):
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")

    # 加载模型
    print(f"正在加载模型: {model_path}")

    # 允许加载自定义类
    with torch.serialization.safe_globals([UnitAwareTransformer]):
        try:
            # 首先尝试使用 weights_only=False
            model = torch.load(model_path, map_location=device, weights_only=False)
        except (TypeError, AttributeError):
            # 如果是旧版本 PyTorch，使用默认参数
            model = torch.load(model_path, map_location=device)

    model.eval()

    # 加载数据集
    print(f"正在加载数据集: {data_file}")
    dataset = ArknightsDataset(data_file, data_enhance=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 进行预测
    correct = 0
    total = 0

    print("开始预测...")
    with torch.no_grad():
        for ls, lc, rs, rc, labels in data_loader:
            ls, lc, rs, rc, labels = [x.to(device) for x in (ls, lc, rs, rc, labels)]

            outputs = model(ls, lc, rs, rc).squeeze()
            preds = (outputs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 打印进度
            if total % (10 * batch_size) == 0:
                print(f"已处理 {total} 条数据...")

    accuracy = 100 * correct / total
    print("\n测试结果:")
    print(f"总数据量: {total}")
    print(f"正确预测: {correct}")
    print(f"准确率: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    # 配置参数
    model_path = "models/best_model_full.pth"  # 使用保存的最佳模型
    data_file = "Z:/66kfpdd.csv"  # 数据文件路径

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        exit(1)

    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        exit(1)

    # 运行测试
    test_model(model_path, data_file)
