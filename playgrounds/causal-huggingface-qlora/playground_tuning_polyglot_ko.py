"""finetune_causal_qlora.py

Fine-tuning of gigantic LLM with the power of QLoRA.

Refer
    https://colab.research.google.com/drive/12albqRjQO7Th-d60W88G1O9y3ED5TQEh
Author: Gyu-min Lee
his.nigel at gmail dot com
"""

import json
import argparse

from pprint import pprint
from datetime import datetime

import yaml
import torch

from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import pipeline
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

bnbConfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT,
                                             quantization_config=bnbConfig,
                                             device_map='auto')

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

TEST = [
    ("대한민국의 수도는 어디인가요?"),
    ("카페라뗴와 카페오레의 차이점은 무엇인가요?"),
    ("영일만친구들은 코딩을 잘 하나요?"),
]

pipe = pipeline("text-generation",
                model=model, tokenizer=tokenizer,
                device_map='auto')

result = [i for i in pipe(i[0] for i in TEST)]
result

dataset = load_dataset(DATASET)

pprint(dataset)

dataset['train'][0]

dataset = dataset.map(lambda x: {'text': f"### 질문: {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>" })
dataset = dataset.map(lambda d: tokenizer(d['text']), batched=True)
dataset['train'][0]['text']

tokenizer.pad_token = tokenizer.eos_token

trainer = Trainer(
    model=model,
    train_dataset=dataset['train'],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        output_dir="ckpts",
        optim="paged_adamw_8bit"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.eval()
model.config.use_cache = True

since = datetime.now()
generated = [model.generate(**tokenizer(t, return_tensors='pt', return_token_type_ids=False),
                max_new_tokens=256,
                early_stopping=True,
                do_sample=True,
                eos_token_id=2,
                ) for t in TEST]
elapsed = (datetime.now()-since).seconds
generated = [tokenizer.decode(x[0]) for x in generated]
for i, j in zip(TEST, generated):
    print("IN: ", end="")
    print(i)
    print("OUT: ", end="")
    print(j)
print(f"Elapsed: {elapsed:04} seconds")

model.save_pretrained('./ckpts/sample_peft')

model.push_to_hub('quantizedCausalTest00')

config = PeftConfig.from_pretrained('gyulukeyi/quantizedCausalTest00')

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model2 = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                              quantization_config=bnb_config, device_map="auto",)
model2 = PeftModel.from_pretrained(model, 'gyulukeyi/quantizedCausalTest00', )
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,)

model2.eval()

since = datetime.now()
generated = [model2.generate(**tokenizer(t, return_tensors='pt', return_token_type_ids=False),
                max_new_tokens=256,
                early_stopping=True,
                do_sample=True,
                eos_token_id=2,
                ) for t in TEST]
elapsed = (datetime.now()-since).seconds
generated = [tokenizer.decode(x[0]) for x in generated]
for i, j in zip(TEST, generated):
    print("IN: ", end="")
    print(i)
    print("OUT: ", end="")
    print(j)
print(f"Elapsed: {elapsed:04} seconds")
