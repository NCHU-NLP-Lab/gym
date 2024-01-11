"""
Preparing
"""

from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import logging
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import math
from accelerate import Accelerator
from transformers import (
    AdamW,
    default_data_collator,
    get_scheduler
)
import wandb


"""
download the model
"""

# # Load model and tokenizer
dataset = load_dataset("glue", "qnli")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)


"""
data preprocessing
"""

# dataset -> DataFrame -> list

data_pre = {}

key_total = dataset.keys()
feature_total = list(dataset['train'].features)[:-1]
for key in key_total:
  data_pre[key] = {}
  for feature in feature_total:
    data_pre[key][feature] = pd.DataFrame(dataset[key]).drop(columns = ['idx'])[feature].tolist()

# tokenizing

train_encoding_total = tokenizer(
    data_pre['train']['question'],
    data_pre['train']['sentence'],
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors="pt"
    )

validation_encoding_total = tokenizer(
    data_pre['validation']['question'],
    data_pre['validation']['sentence'],
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors="pt"
    )

# 加入 Label

def add_targets(encodings, label):
  encodings.update({'label': label})

add_targets(train_encoding_total, data_pre['train']['label'])
add_targets(validation_encoding_total, data_pre['validation']['label'])


"""
定義 Dataset，並轉換成 tensor 格式
"""

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)

train_dataset = Dataset(train_encoding_total)
validation_dataset = Dataset(validation_encoding_total)


"""
準備訓練
"""

# 設定 epoch 與 batch size
train_batch_size = 8      # 設定 training batch size
eval_batch_size = 10      # 設定 eval batch size
num_train_epochs = 3      # 設定 epoch

# 將資料丟入 DataLoader
data_collator = default_data_collator
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch_size)
validation_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=eval_batch_size)

# Optimizer 、Learning rate 、Scheduler 設定
learning_rate=3e-5          # 設定 learning_rate
gradient_accumulation_steps = 1   # 設定 幾步後進行反向傳播
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

# Scheduler and math around the number of training steps.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = num_train_epochs * num_update_steps_per_epoch
print('max_train_steps', max_train_steps)

# scheduler
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=max_train_steps,
)


"""
將資料、參數丟入 Accelerator
"""

# Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
accelerator = Accelerator()

# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, validation_dataloader
)


"""
設定 metric 方法, log console
"""

# Get the metric function
metric = load_metric("glue", "qnli")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state)

total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {max_train_steps}")


"""
w&b
"""

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="qnli",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": num_train_epochs,
    })



"""
train
"""

completed_steps = 0
best_epoch = {"epoch": 0, "acc": 0 }
output_dir = '/user_data/BERT-base/model_qnli'  # your folder

for epoch in trange(num_train_epochs, desc="Epoch"):
  model.train()
  for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    outputs = model(**batch)
    loss = outputs.loss
    loss = loss / gradient_accumulation_steps
    wandb.log({"loss": loss})
    if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      completed_steps += 1

    if step % 50 == 0:
      print({'epoch': epoch, 'step': step, 'loss': loss.item()})

    if completed_steps >= max_train_steps:
      break

  logger.info("***** Running eval *****")
  model.eval()
  for step, batch in enumerate(tqdm(eval_dataloader, desc="Eval Iteration")):
    outputs = model(**batch)
    loss = outputs.loss
    predictions = outputs.logits.argmax(dim=-1)
    metric.add_batch(
        predictions=accelerator.gather(predictions),
        references=accelerator.gather(batch["labels"]),
    )

  eval_metric = metric.compute()
  logger.info(f"epoch {epoch}: {eval_metric}")
  if eval_metric['accuracy'] > best_epoch['acc']:
    best_epoch.update({"epoch": epoch, "acc": eval_metric['accuracy']})
  wandb.log({"accuracy": eval_metric['accuracy']})

  if output_dir is not None:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(output_dir + '/' + '0257_tbs08-ebs10-nte03_ttl_epoch_' + str(epoch), save_function=accelerator.save)

# train_batch_size =8      # 設定 training batch size
# eval_batch_size = 10      # 設定 eval batch size
# num_train_epochs =3      # 設定 epoch