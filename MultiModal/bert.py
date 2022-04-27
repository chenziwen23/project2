# -*- coding: utf-8 -*- 
"""
@Author : Chan ZiWen
@Date : 2022/4/26 14:22
File Description:

Bi-directional LSTM : 主要限制之一是其顺序性，这使得并行训练非常困难
"""
import transformers
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

max_length_test = 20
test_sentence = "曝梅西已通知巴萨他想离开"


def encode_plus():
    # add special tokens
    test_sentence_with_special_tokens = '[CLS]' + test_sentence + '[SEP]'
    tokenized = tokenizer.tokenize(test_sentence_with_special_tokens)

    # convert tokens to ids in WordPiece
    input_ids = tokenizer.convert_tokens_to_ids(tokenized)

    # pre-calculation of pad length, so that we can reuse it later on
    padding_length = max_length_test - len(input_ids)

    # map tokens to WordPiece dictionary and add pad token for those text shorter than our max length
    input_ids = input_ids + ([0] * padding_length)

    # attention should focus dictionary and add pad token for those text shorter than our max length
    attention_mask = [1] * len(input_ids)

    # do not focus just on sequence with non-padded tokens
    attention_mask = attention_mask + ([0] * padding_length)

    # token types needed for example for question answering, for our purpose we will just set 0 as we have just one
    # sequence
    token_type_ids = [0] * max_length_test
    bert_input = {
        "token_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask
    }
    return bert_input


# bert_input = tokenizer.encode_plus(
#     test_sentence,
#     add_special_tokens=True,
#     truncation=True,
#     max_length=max_length_test,
#     padding='max_length',
#     return_attention_mask=True,
# )
#
# print('encoded \n', bert_input)


def split_dataset(df):
    train_set, x = train_test_split(df,
                                    stratify=df['label'],
                                    train_size=0.9,
                                    random_state=42)

    val_set, test_set = train_test_split(x,
                                         test_size=0.5,
                                         random_state=43)
    return train_set, val_set, test_set


df_raw = pd.read_csv("/students/julyedu_529223/data.txt", sep='\t', header=None, names=['text', 'label'])
#  label
df_label = pd.DataFrame({'label': ["财经","房产","股票","教育","科技","社会","时政","体育","游戏","娱乐"],
                         "y": list(range(10))})
df_raw = pd.merge(df_raw, df_label, on='label', how='left')

train_data, val_data, test_data = split_dataset(df_raw)


def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,
                                 truncation=True,
                                 max_length=max_length_test,
                                 padding='max_length',
                                 return_attention_mask=True,
                                 )


# map to the expected input to TFBertForSequenceClassification, see here
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label


def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    if limit > 0:
        ds = ds.take(limit)

    for index, row in ds.iterrows():
        review = row['text']
        label = row['y']
        bert_input = convert_example_to_feature(review)

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(label)

    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)
    ).map(map_example_to_dict)


batch_size = 256
# train dataset
ds_train_encoded = encode_examples(train_data).shuffle(10000).batch(batch_size)

#
ds_val_encoded = encode_examples(val_data).batch(batch_size)

#
ds_test_encoded = encode_examples(test_data).batch(batch_size)

learning_rate = 2e-5
number_of_epochs = 8
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=10)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1)

# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# 修改了 row 727 and 750 of transformers.modeling_tf_utils.py
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)

model.evaluate(ds_test_encoded)

