import os
import random

import pandas as pd
import torch
import torch.serialization
from torch.utils.data import DataLoader, Subset

from train import ArknightsDataset, UnitAwareTransformer


# Placeholder get_device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


# --- End of Placeholder section ---


def load_model_for_inference(model_path, device):
    """Loads a single model for inference."""
    print(f"正在加载模型: {model_path}")
    # Allow loading custom classes
    with torch.serialization.safe_globals(
        [UnitAwareTransformer]
    ):  # Add any other custom classes here
        try:
            # Try with weights_only=False first for custom classes
            model = torch.load(model_path, map_location=device, weights_only=False)
        except (
            TypeError,
            AttributeError,
            RuntimeError,
        ) as e:  # Added RuntimeError for potential unpickling issues
            print(f"加载模型时使用 weights_only=False 失败 ({e}), 尝试默认参数...")
            try:
                model = torch.load(model_path, map_location=device)
            except Exception as e2:
                print(f"使用默认参数加载模型也失败: {e2}")
                raise
    model.eval()
    return model


def compare_models_on_samples(
    model_paths_dict,
    data_file,
    num_samples=5,
    sample_indices=None,
    batch_size_for_sampling=1,  # Process one sample at a time for individual results
):
    """
    Compares multiple models on sampled data from a CSV file.

    Args:
        model_paths_dict (dict): Dictionary where keys are model names (str)
                                 and values are paths to model files (str).
        data_file (str): Path to the CSV data file.
        num_samples (int): Number of samples to randomly draw if sample_indices is None.
        sample_indices (list, optional): A list of specific dataset indices to test.
                                         Overrides num_samples if provided.
        batch_size_for_sampling (int): Should typically be 1 to get individual outputs.
    """
    device = get_device()
    print(f"使用设备: {device}")

    # Load dataset
    print(f"正在加载数据集: {data_file}")
    full_dataset = ArknightsDataset(data_file, data_enhance=False)

    if not full_dataset:
        print("错误: 数据集为空。")
        return

    if sample_indices:
        # Ensure indices are valid
        sample_indices = [idx for idx in sample_indices if idx < len(full_dataset)]
        if not sample_indices:
            print("错误:提供的 sample_indices 无效或超出数据集范围。")
            return
        print(f"使用提供的样本索引: {sample_indices}")
    else:
        if num_samples > len(full_dataset):
            print(
                f"警告: 请求的样本数量 ({num_samples}) 大于数据集大小 ({len(full_dataset)})。将使用所有数据。"
            )
            num_samples = len(full_dataset)
        if num_samples == 0:
            print("错误: num_samples 为 0，无法进行采样。")
            return
        sample_indices = random.sample(range(len(full_dataset)), num_samples)
        print(f"随机抽取 {num_samples} 个样本，索引为: {sample_indices}")

    # Create a Subset and DataLoader for the selected samples
    subset_dataset = Subset(full_dataset, sample_indices)
    # We process one sample at a time to easily map outputs
    data_loader = DataLoader(
        subset_dataset, batch_size=batch_size_for_sampling, shuffle=False
    )

    # Load models
    models = {}
    for model_name, model_path in model_paths_dict.items():
        if not os.path.exists(model_path):
            print(
                f"警告: 模型文件 {model_path} (用于 {model_name}) 不存在，将跳过此模型。"
            )
            continue
        try:
            models[model_name] = load_model_for_inference(model_path, device)
        except Exception as e:
            print(f"加载模型 {model_name} 从 {model_path} 失败: {e}")
            continue

    if not models:
        print("错误: 没有成功加载任何模型。")
        return

    results_list = []
    original_indices_iter = iter(
        sample_indices
    )  # To map back to original dataset index

    print("\n开始在选定样本上进行预测...")
    with torch.no_grad():
        for i, (ls, lc, rs, rc, labels) in enumerate(data_loader):
            # Assuming batch_size_for_sampling is 1 for simplicity here
            # If batch_size_for_sampling > 1, you'd need to iterate through the batch

            current_original_idx = next(original_indices_iter)
            sample_results: dict[str, float] = {
                "Sample Index (Original)": current_original_idx
            }

            ls, lc, rs, rc = [x.to(device) for x in (ls, lc, rs, rc)]

            for model_name, model in models.items():
                outputs = model(ls, lc, rs, rc).squeeze()
                # Ensure outputs is a scalar float; if it's a tensor, get .item()
                if isinstance(outputs, torch.Tensor):
                    # If batch_size_for_sampling > 1, outputs might be a 1D tensor of [batch_size]
                    # For batch_size=1, squeeze() should make it 0-dim or 1-dim with 1 element
                    if outputs.numel() == 1:
                        score = outputs.item()
                    else:
                        # This case should ideally not happen if batch_size_for_sampling=1
                        # Or if it does, you need to decide how to handle multiple outputs for one model call
                        print(
                            f"警告: 模型 {model_name} 对样本 {current_original_idx} 的输出不是标量: {outputs}"
                        )
                        score = float("nan")  # Or handle as a list/array
                else:
                    score = float(outputs)  # If it's already a Python float

                sample_results[model_name] = score

            results_list.append(sample_results)

    # Display results in a table
    if results_list:
        results_df = pd.DataFrame(results_list)
        # Set Sample Index as the DataFrame index
        results_df.set_index("Sample Index (Original)", inplace=True)
        # 将数值列格式化为两位小数
        results_df = results_df.round(2)
        print("\n多模型横向对比结果:")
        print(results_df)
    else:
        print("没有生成任何结果。")


if __name__ == "__main__":
    model_paths_config = {
        "Model_89": "Z:/best_model_full_66kpure_89.51.pth",
        "Model_92": "Z:/best_model_full_6345ywz3_92.pth",
        "Model_93": "models/best_model_full.pth",
    }

    # 数据文件路径 (使用您的实际测试集CSV)
    # For demonstration, we'll use a dummy name. The ArknightsDataset placeholder doesn't actually read it.
    data_csv_file = "6345ywz3(测试集).csv"

    if not os.path.exists(data_csv_file):
        print(f"错误: 数据文件不存在: {data_csv_file}")
        exit(1)

    # 模型文件存在性检查在 compare_models_on_samples 函数内部处理

    # --- 运行横向对比 ---
    print("开始多模型横向对比...")
    compare_models_on_samples(
        model_paths_dict=model_paths_config,
        data_file=data_csv_file,
        num_samples=15,  # 从CSV中随机抽取3条数据进行对比
        # sample_indices=[0, 10, 25] # 或者，指定特定的数据行索引 (从0开始)
    )
