# CoLA Sentence Classification with BERT

* Model: BERT (Base)
* Dataset: The Corpus of Linguistic Acceptability (CoLA) from GLUE

三種版本如下

| Name                 | Model Class                                                  | Training Method                                              |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Transformers Trainer | [transformers.BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification) | [transformers.Trainer](https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer) |
| Transformers Scratch | [transformers.BertForSequenceClassification](https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification) | PyTorch training loop                                        |
| Pytorch Scratch      | Custom PyTorch [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | PyTorch Training Loop                                        |

## Reference

Code are mainly adapted from the following sources:

* [HuggingFace Transformers Example](https://github.com/huggingface/transformers/tree/v4.5.1/examples/text-classification)
* [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/)
* [Source Code of BertForSequenceClassification](https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForSequenceClassification)

## Acknowledgement

### BERT

```bibtex
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### GLUE

```
@unpublished{wang2018glue
    title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for
            Natural Language Understanding}
    author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill,
            Felix and Levy, Omer and Bowman, Samuel R.}
    note={arXiv preprint 1804.07461}
    year={2018}
}
```

