from transformers import AutoTokenizer
import torch

def QA_model(model, context, question):

  tokenizer=AutoTokenizer.from_pretrained('bert-base-chinese')
  input = tokenizer.encode_plus(question, context, return_tensors='pt')
  input.to('cuda')
  outputs = model(**input)
  answer_start = torch.argmax(outputs[0])
  answer_end = torch.argmax(outputs[1]) + 1

  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input['input_ids'][0][answer_start:answer_end]))


  return answer

model = torch.load("model",map_location=torch.device('cuda'))
context = "《哈利波特》，英國作家J·K·羅琳的奇幻文學系列小說，描寫主角哈利波特在霍格華茲魔法學校7年學習生活中的冒險故事；該系列被翻譯成77種語言，在超過兩百個國家出版，所有版本的總銷售量5億，名列世界上最暢銷小說之列。[1][2][3]美國華納兄弟電影公司把這7集小說改拍成8部電影，前6集各一部，第7集分兩部。哈利波特電影系列是全球史上最賣座的電影系列，總票房收入過77億美元，在多國上映。[4]此外還有位於佛羅里達的主題樂園，與大量衍生作品，使作者羅琳本人在此作品完成後躋身英國最富裕的女性人物之列，超越英國女王伊莉莎白二世。"
question = "哈利波特的作家是誰?"

answer = QA_model(model, context, question)  
print('\ncontext:',context,'\n')
print('question:',question,'\n')
print('answer:',answer,'\n')
# print(answer)