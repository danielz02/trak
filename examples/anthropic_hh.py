# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, default_data_collator

from trl import RewardConfig, RewardTrainer, is_xpu_available

from trak import TRAKer
from trak.contrib.reward_model import RLHFRewardModelingOutput

tqdm.pandas()


class DebertaRewardModelForTrakAttribution(nn.Module):
    """
    Wrapper for HuggingFace sequence classification models.
    """

    def __init__(self, model_id: str, **kwargs):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            ignore_mismatched_sizes=False,
            **kwargs
        )

        self.model.eval().cuda()

    def forward(self, input_ids_chosen, token_type_ids_chosen, attention_mask_chosen, input_ids_rejected,
                token_type_ids_rejected, attention_mask_rejected):
        rewards_chosen = self.model(
            input_ids=input_ids_chosen, token_type_ids=token_type_ids_chosen, attention_mask=attention_mask_chosen,
        )[0]

        rewards_rejected = self.model(
            input_ids=input_ids_rejected, token_type_ids=token_type_ids_rejected, attention_mask=attention_mask_rejected,
        )[0]

        return rewards_chosen - rewards_rejected


@dataclass
class ScriptArguments:
    model_name: str = "facebook/opt-350m"
    """the model name"""
    dataset_name: str = "Anthropic/hh-rlhf"
    """the dataset name"""
    dataset_text_field: str = "text"
    """the text field of the dataset"""
    eval_split: str = "none"
    """the dataset split to evaluate on; default to 'none' (no evaluation)"""
    load_in_8bit: bool = False
    """load the model in 8 bits precision"""
    load_in_4bit: bool = False
    """load the model in 4 bits precision"""
    trust_remote_code: bool = True
    """Enable `trust_remote_code`"""
    reward_config: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            output_dir="output",
            per_device_train_batch_size=64,
            num_train_epochs=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1.41e-5,
            report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            logging_steps=500,
            evaluation_strategy="no",
            max_length=512,
        )
    )
    use_peft: bool = False
    """whether to use peft"""
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        ),
    )

    train_size: int = 50_000
    val_size: int = 5_463
    out: str = "./results"


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "token_type_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "token_type_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        if "token_type_ids" in tokenized_chosen:
            if "token_type_ids_chosen" not in new_examples:
                new_examples["token_type_ids_chosen"] = []
                new_examples["token_type_ids_rejected"] = []
            new_examples["token_type_ids_chosen"].append(tokenized_chosen["token_type_ids"])
            new_examples["token_type_ids_rejected"].append(tokenized_rejected["token_type_ids"])

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


def train():
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        num_labels=1,
    )

    # Step 5: Define the Trainer
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args.reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()


def process_batch(batch):
    return (batch['input_ids_chosen'], batch['attention_mask_chosen'], batch['token_type_ids_chosen'],
            batch['input_ids_rejected'], batch['attention_mask_rejected'], batch['token_type_ids_rejected'])


def init_loaders(ds_train, ds_val, batch_size=16):
    ds_train = ds_train.select(range(args.train_size))
    ds_val = ds_val.select(range(args.val_size))
    return DataLoader(ds_train, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator), \
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)


def attribute():
    device = 'cuda'
    loader_train, loader_val = init_loaders(train_dataset, eval_dataset)
    model = DebertaRewardModelForTrakAttribution(args.model_name)

    traker = TRAKer(model=model,
                    task=RLHFRewardModelingOutput,
                    train_set_size=args.train_size,
                    save_dir=args.out,
                    device=device,
                    proj_dim=1024)

    traker.load_checkpoint(model.state_dict(), model_id=0)
    for batch in tqdm(loader_train, desc='Featurizing..'):
        # process batch into compatible form for TRAKer TextClassificationModelOutput
        batch = process_batch(batch)
        batch = [x.cuda() for x in batch]
        traker.featurize(batch=batch, num_samples=batch[0].shape[0])

    traker.finalize_features()

    traker.start_scoring_checkpoint(exp_name='hh-rlhf',
                                    checkpoint=model.state_dict(),
                                    model_id=0,
                                    num_targets=args.val_size)
    for batch in tqdm(loader_val, desc='Scoring..'):
        batch = process_batch(batch)
        batch = [x.cuda() for x in batch]
        traker.score(batch=batch, num_samples=batch[0].shape[0])

    scores = traker.finalize_scores(exp_name='hh-rlhf')


if __name__ == "__main__":
    args = tyro.cli(ScriptArguments)
    args.reward_config.evaluation_strategy = "steps" if args.eval_split != "none" else "no"

    # Step 1: Load the model
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
        # Copy the model to each device
        device_map = (
            {"": f"xpu:{Accelerator().local_process_index}"}
            if is_xpu_available()
            else {"": Accelerator().local_process_index}
        )
    else:
        device_map = None
        quantization_config = None

    # Step 2: Load the dataset and pre-process it
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = load_dataset(args.dataset_name, split="train")

    if args.eval_split == "none":
        eval_dataset = None
    else:
        eval_dataset = load_dataset(args.dataset_name, split=args.eval_split)

        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length
                      and len(x["input_ids_rejected"]) <= args.reward_config.max_length
        )

    # Step 4: Define the LoraConfig
    if args.use_peft:
        peft_config = args.peft_config
    else:
        peft_config = None

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.reward_config.max_length and
                  len(x["input_ids_rejected"]) <= args.reward_config.max_length
    )

    attribute()
