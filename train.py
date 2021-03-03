#! -*- coding: utf-8 -*-
# 词级别的中文PEGASUS预训练

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.tokenizers import load_vocab, save_vocab
from bert4keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import text_segmentate
import pylcs
import jieba
jieba.initialize()

# 基本参数
maxlen = 512
batch_size = 96
epochs = 100000
summary_rate = 0.25
t_maxlen = maxlen // 4
s_maxlen = maxlen - t_maxlen

# T5配置
config_path = '/root/kg/bert/mt5/mt5_base/mt5_base_config.json'
checkpoint_path = '/root/kg/bert/mt5/mt5_base/model.ckpt-1000000'
spm_path = '/root/kg/bert/mt5/sentencepiece.model'

# PEGASUS
dict_path_1 = '/root/kg/bert/chinese_pegasus_L-12_H-768_A-12/vocab.txt'
dict_path_2 = '/root/kg/bert/chinese_t5_pegasus_base/vocab.txt'

# 构建词表
sp_tokenizer = SpTokenizer(spm_path, token_start=None, token_end=None)
token_dict = load_vocab(dict_path_1)
keep_tokens, new_token_dict, n = [], {}, 0
for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
    if n < 106:
        new_token_dict[t] = n
        n += 1
        continue
    if t.startswith('##'):
        i = sp_tokenizer.token_to_id(t[2:])
        if i == 2:
            i = sp_tokenizer.token_to_id(u'\u2581' + t)
    else:
        i = sp_tokenizer.token_to_id(u'\u2581' + t)
        if i == 2:
            i = sp_tokenizer.token_to_id(t)
    if i != 2:
        keep_tokens.append(i)
        new_token_dict[t] = len(new_token_dict)

keep_tokens = [2] * 106 + keep_tokens
keep_tokens_inv = {j: i for i, j in enumerate(keep_tokens)}

compound_tokens = []
for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
    if t not in new_token_dict:
        new_token_dict[t] = len(new_token_dict)
        ids = [keep_tokens_inv.get(i, 0) for i in sp_tokenizer.encode(t)[0]]
        compound_tokens.append(ids)

save_vocab(dict_path_2, new_token_dict)

# 构建分词器
tokenizer = Tokenizer(
    new_token_dict,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


def corpus():
    """语料生成器
    """
    while True:
        f = '/root/data_pretrain/data_shuf.json'
        with open(f) as f:
            for l in f:
                l = json.loads(l)
                for texts in text_process(l['text']):
                    yield texts


def text_process(text):
    """分割文本
    """
    texts = text_segmentate(text, 32, u'\n。')
    result, length = [], 0
    for text in texts:
        if length + len(text) > maxlen * 1.5 and len(result) >= 3:
            yield result
            result, length = [], 0
        result.append(text)
        length += len(text)
    if result and len(result) >= 3:
        yield result


def gather_join(texts, idxs):
    """取出对应的text，然后拼接起来
    """
    return ''.join([texts[i] for i in idxs])


def pseudo_summary(texts):
    """构建伪标签摘要数据集
    """
    source_idxs, target_idxs = list(range(len(texts))), []
    while True:
        sims = []
        for i in source_idxs:
            new_source_idxs = [j for j in source_idxs if j != i]
            new_target_idxs = sorted(target_idxs + [i])
            new_source = gather_join(texts, new_source_idxs)
            new_target = gather_join(texts, new_target_idxs)
            sim = pylcs.lcs(new_source, new_target)
            sims.append(sim)
        new_idx = source_idxs[np.argmax(sims)]
        source_idxs.remove(new_idx)
        target_idxs = sorted(target_idxs + [new_idx])
        source = gather_join(texts, source_idxs)
        target = gather_join(texts, target_idxs)
        if (
            len(source_idxs) == 1 or
            1.0 * len(target) / len(source) > summary_rate
        ):
            break
    if len(source) < len(target):
        source, target = target, source
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        for is_end, texts in self.sample(random):
            source, target = pseudo_summary(texts)
            source_ids, _ = tokenizer.encode(source, maxlen=s_maxlen)
            target_ids, _ = tokenizer.encode(target, maxlen=t_maxlen)
            yield source_ids, target_ids


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * y_mask) / K.sum(y_mask)
        self.add_metric(acc, name='accuracy', aggregation='mean')
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    t5 = build_transformer_model(
        config_path,
        checkpoint_path=None,
        model='t5.1.1',
        with_lm='linear',
        keep_tokens=keep_tokens,
        compound_tokens=compound_tokens,
        return_keras_model=False,
    )

    model = t5.model
    output = CrossEntropy(1)(model.inputs[1:] + model.outputs)
    model = keras.models.Model(model.inputs, output)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    optimizer = AdamWLR(
        learning_rate=1e-4,
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        lr_schedule={10000: 1}
    )
    model.compile(optimizer=optimizer)
    model.summary()
    t5.load_weights_from_checkpoint(checkpoint_path)


class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('t5_pegasus_model.weights')  # 保存模型


if __name__ == '__main__':

    # 启动训练
    evaluator = Evaluator()
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32'),
        shapes=([None], [None]),
        names=('Encoder-Input-Token', 'Decoder-Input-Token'),
        padded_batch=True
    )

    model.fit(
        dataset, steps_per_epoch=1000, epochs=epochs, callbacks=[evaluator]
    )

else:

    model.load_weights('t5_pegasus_model.weights')
