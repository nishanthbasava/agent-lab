# Small project where I'm running SFT of Qwen on the multi-turn Capybara dataset using trl from hugging face.


from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

#smallest instruction-tuned model in alibaba's qwen2.5 family so it doesn't cost too much to fine-tune
MODEL = "Qwen/Qwen2.5-0.5B-Instruct" 

def main():
    # Capybara in conversational format ({"messages": [...]})
    dataset = load_dataset("trl-lib/Capybara", split="train")

    config = SFTConfig(
        output_dir="qwen-capybara-sft",
        per_device_train_batch_size=2, #batch training with size 2
        gradient_accumulation_steps=4, 
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        max_length=1024,
        bf16=True,
    )

    # a lot of abstraction handled here. will dig deeper and implement later.
    trainer = SFTTrainer(
        model=MODEL,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
