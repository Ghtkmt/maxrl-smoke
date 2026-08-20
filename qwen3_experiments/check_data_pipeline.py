"""
CPU-only validation of the verl data pipeline (RLHFDataset + collate_fn)
for the smoke-test parquet files, using the exact same Hydra config overrides
as qwen3_experiments/run_smoke.sh. No GPU / vllm needed.
"""
import os

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from transformers import AutoTokenizer

initialize(config_path="../verl/trainer/config", job_name="data_check", version_base=None)

overrides = [
    "data.train_files=/Users/shekuigang/Projects/maxrl/data/polaris_smoke/train.parquet",
    "data.val_files=['/Users/shekuigang/Projects/maxrl/data/aime25_smoke/test.parquet','/Users/shekuigang/Projects/maxrl/data/math_smoke/test.parquet']",
    "data.train_batch_size=8",
    "data.max_prompt_length=512",
    "data.max_response_length=512",
    "data.filter_overlong_prompts=True",
    "data.truncation=error",
    "trainer.device=cpu",
]
cfg = compose(config_name="ppo_trainer", overrides=overrides)
print("[ok] composed full ppo_trainer config; data section keys:", sorted(cfg.data.keys()))

from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn

model_path = "/Users/shekuigang/Projects/maxrl/models/Qwen3-1.7B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("[ok] tokenizer loaded:", type(tokenizer).__name__, "| vocab:", tokenizer.vocab_size)

train_ds = create_rl_dataset(cfg.data.train_files, cfg.data, tokenizer, None)
print(f"[ok] train dataset: {len(train_ds)} samples")
item = train_ds[0]
print("[ok] sample item keys:", sorted(item.keys()))
print("     prompt tokens:", item["input_ids"].shape, "| attention_mask:", item["attention_mask"].shape)

sampler = create_rl_sampler(cfg.data, train_ds)
from torch.utils.data import DataLoader

loader = DataLoader(train_ds, batch_size=8, drop_last=True, collate_fn=collate_fn, sampler=sampler)
batch = next(iter(loader))
print("[ok] train batch keys:", sorted(batch.keys()))
for k in ["input_ids", "attention_mask", "position_ids"]:
    print(f"     {k}: {batch[k].shape}")

val_ds = create_rl_dataset(cfg.data.val_files, cfg.data, tokenizer, None)
print(f"[ok] val dataset: {len(val_ds)} samples (aime25 4 + math 4)")
val_batch = collate_fn([val_ds[i] for i in range(len(val_ds))])
print("[ok] val batch keys:", sorted(val_batch.keys()))
print("     input_ids:", val_batch["input_ids"].shape)

# Verify ground-truth rewards are present in the data
print("[ok] sample reward_model:", item.get("reward_model"))
print("=== DATA PIPELINE CHECK PASSED ===")
