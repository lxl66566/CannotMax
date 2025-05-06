import csv
import zipfile
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import IO


def process_csv_file(csv_file: IO[bytes], filename=""):
    """处理单个 CSV 文件

    Args:
        csv_file: 文件对象
        filename: 文件名，用于错误报告

    Returns:
        list: 处理后的数据行列表
    """
    processed_data = []
    reader = csv.reader(TextIOWrapper(csv_file, "utf-8"))
    next(reader)  # 跳过首行
    for row in reader:
        try:
            if len(row) >= 113:
                processed_row = [
                    int(float(x)) if i < 112 else x for i, x in enumerate(row[:113])
                ]
                processed_data.append(processed_row)
        except Exception as e:
            print(f"错误: {e}")
            print(f"行: {row}")
            print(f"文件: {filename}")
            continue
    return processed_data


def process_zip_files(input_folder):
    """合并 zip 里的 csv 数据"""
    all_data = []

    for zip_path in Path(input_folder).glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith(".csv"):
                        with zip_ref.open(file_info) as csv_file:
                            processed_data = process_csv_file(
                                csv_file, f"{zip_path}/{file_info.filename}"
                            )
                            all_data.extend(processed_data)

            print(f"处理完成: {zip_path}")
        except Exception as e:
            print(f"错误: {e}")
            print(f"文件: {zip_path}")

    for csv_path in Path(input_folder).glob("*.csv"):
        try:
            with open(csv_path, "rb") as f:
                processed_data = process_csv_file(f, str(csv_path))
                all_data.extend(processed_data)
        except Exception as e:
            print(f"错误: {e}")
            print(f"文件: {csv_path}")

        print(f"处理完成: {csv_path}")
    return all_data


def deduplicate_data(data):
    """去重"""
    groups = defaultdict(list)
    for row in data:
        key = tuple(row[:112])
        groups[key].append(row[112])

    result = []
    for key, values in groups.items():
        if len(values) == 1:
            result.append(list(key) + [values[0]])
        else:
            if all(v == values[0] for v in values):
                result.append(list(key) + [values[0]])
            else:
                count = defaultdict(int)
                for v in values:
                    count[v] += 1
                max_count = max(count.values())
                candidates = [v for v, c in count.items() if c == max_count]
                if len(candidates) == 1:
                    result.append(list(key) + [candidates[0]])
    return result


def save_to_csv(data, output_file="merged_arknights.csv"):
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def preprocess(input_folder):
    # 直接处理ZIP文件，避免解压覆盖
    data = process_zip_files(input_folder)

    if not data:
        print("警告：未找到任何有效的 csv 数据")
        return

    deduplicated_data = deduplicate_data(data)
    save_to_csv(deduplicated_data)

    print(f"处理完成: 原始数据 {len(data)} 条，去重后 {len(deduplicated_data)} 条")
    print("结果已保存到 merged_arknights.csv")


if __name__ == "__main__":
    preprocess("Z:/data")
