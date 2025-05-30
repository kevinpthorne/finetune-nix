import os
import tempfile
import argparse
from bs4 import BeautifulSoup
import requests
from git import Repo
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch

from disk_memoize import disk_memoize

DOC_URLS = [
    "https://nix.dev/manual/nix/2.28/",
    "https://nixos.org/manual/nixpkgs/stable/",
    "https://nixos.org/manual/nixos/stable/",
]

REPO_URL = "https://github.com/NixOS/nixpkgs.git"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-base")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--use-qlora", action="store_true", help="Enable QLoRA + PEFT fine-tuning")
    return parser.parse_args()


@disk_memoize(".cache/docs.pkl")
def scrape_docs(urls):
    print("Scraping documentation...")
    texts = []
    for base_url in urls:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Get all links from the TOC or main content
        links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(base_url)]
        links = list(set(links + [base_url]))  # Add base page too

        for link in links:
            try:
                page = requests.get(link)
                soup = BeautifulSoup(page.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                if len(text) > 200:  # Filter short pages
                    texts.append(text)
            except Exception as e:
                print(f"Failed to scrape {link}: {e}")
    return texts

@disk_memoize(".cache/repo.pkl")
def clone_repo_and_extract(repo_url):
    print("Cloning nixpkgs repo...")
    with tempfile.TemporaryDirectory() as tmpdir:
        Repo.clone_from(repo_url, tmpdir)
        all_code = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(('.nix', '.md', '.sh', '.json')):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content) > 100:
                                all_code.append(content)
                    except Exception as e:
                        continue
        return all_code

@disk_memoize(".cache/tokenized_input.pkl")
def prepare_dataset(texts, tokenizer):
    print("Tokenizing...")
    tokenized = tokenizer(texts, truncation=True, padding='max_length', max_length=512)
    return Dataset.from_dict(tokenized)

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     # Load DeepSeek-Coder model
#     model_name = "deepseek-ai/deepseek-coder-6.7b-base"  # Replace with latest if needed
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

#     # Gather data
#     doc_texts = scrape_docs(DOC_URLS)
#     code_texts = clone_repo_and_extract(REPO_URL)
#     all_texts = doc_texts + code_texts

#     # Tokenize and prepare dataset
#     dataset = prepare_dataset(all_texts, tokenizer)

#     # dump cache
#     torch.cuda.empty_cache()

#     # Define training args
#     training_args = TrainingArguments(
#         output_dir="./deepseek-nix-finetuned",
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=8,
#         num_train_epochs=2,
#         fp16=True,
#         save_steps=500,
#         logging_steps=100,
#         evaluation_strategy="no",
#         report_to="none",
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         tokenizer=tokenizer,
#     )

#     print("Starting training...")
#     trainer.train()
#     model.save_pretrained("./deepseek-nix-finetuned")

# if __name__ == "__main__":
#     main()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model, optionally with 8-bit
    if args.use_qlora:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quant_config,
            trust_remote_code=True,
        ).to(device)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)

    # Gather data
    doc_texts = scrape_docs(DOC_URLS)
    code_texts = clone_repo_and_extract(REPO_URL)
    all_texts = doc_texts + code_texts

    # Tokenize and prepare dataset
    dataset = prepare_dataset(all_texts, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()
    model.save_pretrained("./deepseek-nix-finetuned")

if __name__ == "__main__":
    main()
