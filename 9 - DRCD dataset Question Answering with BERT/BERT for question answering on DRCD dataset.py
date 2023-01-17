from tensorflow import keras
import json
from pprint import pprint
from pathlib import Path
from transformers import AutoTokenizer, BertTokenizerFast,BertForQuestionAnswering, AutoConfig,  AdamW, default_data_collator, get_scheduler
import torch
import logging
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
import math
import transformers
from accelerate import Accelerator
from evaluate import load
import evaluate
import os
# from sklearn.metrics import f1_score

# 讀資料
def read_data(path, limit=None):
    path = Path(path)
    with open(path, 'rb') as f:
        data_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in data_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                # answer = qa['answers']
                for answer in qa['answers']:
                  contexts.append(context)
                  questions.append(question)
                  answers.append(answer)
                if limit != None and len(contexts) > limit:
                  return contexts, questions, answers
                  
    return contexts, questions, answers

# 新增 answer 的結束位置
def add_end_idx(answers):
    for answer in answers:
        gold_text = answer['text']
        start_idx = answer['answer_start']
        if gold_text == '':
          end_idx = 0
        else:
          end_idx = start_idx + len(gold_text) # Find end character index of answer in context
        answer['answer_end'] = end_idx

# 新增答案的起始結束位置
def add_token_positions(encodings,answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    if answers[i]['answer_start'] == 0 and answers[i]['answer_end'] == 0:
      start_positions.append(0)
      end_positions.append(0)
    else:
      start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
      end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
    # if None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = tokenizer.model_max_length
  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# 定義dataset(轉換成tensor格式)
class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)
# ----------------------------------------------------------------------------------

# Download the data
train_data_url = "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/master/DRCD_training.json"
train_path = keras.utils.get_file("train.json", train_data_url)
eval_data_url = "https://github.com/DRCKnowledgeTeam/DRCD/blob/master/DRCD_dev.json?raw=true"
eval_path = keras.utils.get_file("eval.json", eval_data_url)

with open(train_path, "r", encoding="UTF-8") as f:
    raw_train_data = json.load(f)

with open(eval_path, "r", encoding="UTF-8") as f:
    raw_eval_data = json.load(f)
    
# for ele in raw_train_data['data']:
#   pprint(ele['paragraphs'][0])
#   break

# 取訓練資料10000筆，評估資料2000筆
train_contexts, train_questions, train_answers = read_data(train_path, 10000)
eval_contexts, eval_questions, eval_answers = read_data(eval_path,2000)
# print(len(train_contexts))
# print(len(eval_contexts))
# print("-------------------------------------------------------------------------")
# print('train_data')
# print("context:",train_contexts[0])
# print("question:",train_questions[0])
# print("answer:",train_answers[0])
# print("evaluation_data")
# print("context:",eval_contexts[0])
# print("question:",eval_questions[0])
# print("answer:",eval_answers[0])
# print("-")
# print("context:",eval_contexts[1])
# print("question:",eval_questions[1])
# print("answer:",eval_answers[1])

# print("-------------------------------------------------------------------------")
# 新增測試及評估資料的answer結束位置
add_end_idx(train_answers)
add_end_idx(eval_answers)
# print(eval_answers[0])
# print(eval_answers[1])
# print("-------------------------------------------------------------------------")

# 將資料進行 Tokenize
# 將 input 資料轉換成 token id 、tpye_id 與 attention_mask
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
# tokenizer_fast = BertTokenizerFast.from_pretrained('bert-base-chinese')
config = AutoConfig.from_pretrained('bert-base-chinese')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
eval_encodings = tokenizer(eval_contexts, eval_questions, truncation=True, padding=True)

print('tokenizer 最大input長度 = ' ,tokenizer.model_max_length)
train_encodings.keys()
print("-------------------------------------------------------------------------")

# # use AutoTokenizer
# print("input_ids\n" ,train_encodings['input_ids'][0])
# print("length = ",len(train_encodings['input_ids'][0]))
# print("token_type_ids = ",train_encodings['token_type_ids'][0])
# print("attention_mask = ",train_encodings['attention_mask'][0])

# print("\ninput_ids convert_ids_to_tokens \n",tokenizer.convert_ids_to_tokens(train_encodings['input_ids'][0]))
# print("\ninput_ids decode\n", tokenizer.decode(train_encodings['input_ids'][0]))

# print("\nattention_mask\n",train_encodings['attention_mask'][0])
# print("length = ",len(train_encodings['attention_mask'][0]))
# print("-------------------------------------------------------------------------")

# 新增答案的起始結束位置
# print(len(train_answers))
# print(len(eval_answers))
add_token_positions(train_encodings, train_answers)
add_token_positions(eval_encodings, eval_answers)


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#dataset轉換成tensor格式
train_dataset = Dataset(train_encodings)
eval_dataset = Dataset(eval_encodings)
train_dataset[0]
# print('train_dataset長度:',len(train_dataset))
# print("-------------------------------------------------------------------------")


# ----------------------------------------------------------------------------------

# 載入模型架構( BertForQuestionAnswering )
model = BertForQuestionAnswering.from_pretrained('bert-base-chinese', config=config).to(device)
# print(model)
# print(next(model.parameters()).device)  
# model = model.cuda()
  
batch_size = 8
optim = AdamW(model.parameters(), lr=3e-5)
epochs = 5

# ----------------------------------------------------------------------------------
# train
print("train")
train_loader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)
for epoch in range(epochs):
   model.train()
   loop = tqdm(train_loader, leave = True)

   for batch in loop:
      optim.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      start_position = batch['start_positions'].to(device)
      end_position = batch['end_positions'].to(device)

      outputs = model(input_ids, attention_mask = attention_mask, start_positions = start_position, end_positions = end_position)

      loss = outputs[0]

      loss.backward()

      optim.step()

      loop.set_description(f'Epoch {epoch}')
      loop.set_postfix(loss = loss.item()) 
   
# ----------------------------------------------------------------------------------
# evalation
print("evalating")
model.eval()
score = []
pre = []
ref = []
prediction_text = []
answer_text = []
index = 0

eval_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
for batch in eval_dataloader:
  with torch.no_grad():
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_position = batch['start_positions'].to(device)
    end_position = batch['end_positions'].to(device)

    outputs = model(input_ids, attention_mask = attention_mask)
    start_pred = torch.argmax(outputs['start_logits'], dim=1)
    end_pred = torch.argmax(outputs['end_logits'], dim=1)

    for i in range(len(input_ids)):
      # print(tokenizer.decode(input_ids[i]))
      # context=tokenizer.decode(input_ids[i])
      # print('問題:',context[context.find('[SEP]'): context.rfind('[SEP]')])
      pre_text=tokenizer.decode(input_ids[i][start_position[i]: end_position[i]])
      ref_text=tokenizer.decode(input_ids[i][start_pred[i]: end_pred[i]])
      # print('預測:',pre_text)
      # print('答案:',ref_text)
      # print("--------------------------------------")
      prediction_text.append(pre_text)
      answer_text.append(ref_text)
      pre.append({'prediction_text': pre_text, 'id': str(index)})
      ref.append({'answer': {'answer_start': start_pred[i], 'text': ref_text}, 'id': str(index)})

  

    score.append(((start_pred == start_position).sum()/len(start_pred)).item())
    score.append(((end_pred == end_position).sum()/len(end_pred)).item())

    index+=1

score = sum(score)/len(score)
print("score:",score)

torch.save(model,'./model')
# ----------------------------------------------------------------------------------
# EM score
exact_match_metric = load("exact_match")
em_results = exact_match_metric.compute(predictions=prediction_text, references=answer_text)
# print('em:',em_results)
print('em:',em_results["exact_match"])
# print(prediction_text)
# print(answer_text)

# F1 score
# f1_metric = evaluate.load("f1")
# f1_results = f1_metric.compute(predictions=pre, references=ref)
# print('f1:',f1_results)
# ----------------------------------------------------------------------------------
