# SQuAD 2.0 Question Answering with BERT

* Model: bert-base-uncased
* Dataset: Stanford Question Answering Dataset (SQuAD) 2.0

兩種版本如下

| Name         | Model Class   | Training Method  |
|-------|-------|-------|
| bert_qa_train(with_plausible_answers) | bert-base-uncased | BertForQuestionAnswering |
| bert_qa_train(without_plausible_answers) | bert-base-uncased | BertForQuestionAnswering |