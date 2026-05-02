"""
预下载所有数据集到本地，避免每次训练时从 HuggingFace 下载
运行: python download_datasets.py
"""

import os
from datasets import load_dataset

# Optional HuggingFace endpoint override.
# Users in mainland China can speed up downloads by exporting:
#     HF_ENDPOINT=https://hf-mirror.com
# before running this script. We do NOT override an endpoint the user has
# already set in their environment.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

HUGGINGFACE_REPO = "thuml/Time-Series-Library"

# 数据集配置: (dataset_name, save_path)
DATASETS = [
    # ETT 数据集
    ("ETTh1", "./dataset/ETT-small/ETTh1.csv"),
    ("ETTh2", "./dataset/ETT-small/ETTh2.csv"),
    ("ETTm1", "./dataset/ETT-small/ETTm1.csv"),
    ("ETTm2", "./dataset/ETT-small/ETTm2.csv"),
    # 其他数据集
    ("weather", "./dataset/weather/weather.csv"),
    ("electricity", "./dataset/electricity/electricity.csv"),
    ("traffic", "./dataset/traffic/traffic.csv"),
    ("exchange_rate", "./dataset/exchange_rate/exchange_rate.csv"),
    ("solar_energy", "./dataset/solar_energy/solar_energy.csv"),
]


def download_and_save(dataset_name, save_path):
    """下载数据集并保存到本地"""
    if os.path.exists(save_path):
        print(f"[SKIP] {dataset_name} already exists at {save_path}")
        return

    print(f"[DOWNLOAD] {dataset_name} from HuggingFace...")
    try:
        ds = load_dataset(HUGGINGFACE_REPO, name=dataset_name)
        df = ds["train"].to_pandas()

        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存为 CSV
        df.to_csv(save_path, index=False)
        print(f"[SAVED] {dataset_name} -> {save_path} ({len(df)} rows)")
    except Exception as e:
        print(f"[ERROR] Failed to download {dataset_name}: {e}")


def main():
    print("=" * 60)
    print("Downloading datasets from HuggingFace to local...")
    print("=" * 60)

    for dataset_name, save_path in DATASETS:
        download_and_save(dataset_name, save_path)

    print("=" * 60)
    print("Done! All datasets are saved locally.")
    print("=" * 60)


if __name__ == "__main__":
    main()
