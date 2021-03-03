# T5 PEGASUS

中文生成式预训练模型，以mT5为基础架构和初始权重，通过类似PEGASUS的方式进行预训练。

详情可见：https://kexue.fm/archives/8209

## Tokenizer

我们将T5 PEGASUS的Tokenizer换成了BERT的Tokenizer，它对中文更加友好。同时，我们重新整理了一版词表，使得里边的字、词都更加完善，目前的vocab.txt共包含5万个token，真正覆盖了中文的常用字、词。

## 预训练任务

预训练任务模仿了PEGASUS的摘要式预训练。具体来说，假设一个文档有n个句子，我们从中挑出大约n/4个句子（可以不连续），使得这n/4个句子拼起来的文本，跟剩下的3n/4个句子拼起来的文本，最长公共子序列尽可能长，然后我们将3n/4个句子拼起来的文本视为原文，n/4个句子拼起来的文本视为摘要，这样就构成了一个“(原文, 摘要)”的伪摘要数据对了。

## 模型下载

## 部分评测

## 如何引用

Bibtex：

```latex
@techreport{zhuiyit5pegasus,
  title={T5 PEGASUS - ZhuiyiAI},
  author={Jianlin Su},
  year={2021},
  url="https://github.com/ZhuiyiTechnology/t5-pegasus",
}
```

## 联系我们

邮箱：ai@wezhuiyi.com 追一科技：https://zhuiyi.ai








