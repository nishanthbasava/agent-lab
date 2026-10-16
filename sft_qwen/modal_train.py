"""Run the same Qwen SFT on Modal with a GPU.

Usage:
    modal run modal_train.py
"""

import modal

app = modal.App("sft-qwen-capybara")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "datasets",
    "trl",
    "accelerate",
)

# Persist checkpoints across runs
volume = modal.Volume.from_name("sft-qwen-outputs", create_if_missing=True)

@app.function(image=image, gpu="A10G", timeout=60 * 60 * 4, volumes={"/outputs": volume})
def train():
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("trl-lib/Capybara", split="train")

    config = SFTConfig(
        output_dir="/outputs/qwen-capybara-sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        max_length=1024,
        bf16=True,
    )

    trainer = SFTTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        train_dataset=dataset,
        args=config,
    )
    trainer.train()
    trainer.save_model()
    volume.commit()

@app.local_entrypoint()
def main():
    train.remote()
