# import os, glob
# dirname = '/students/julyedu_529223/tiny-imagenet-200'
#
#
#
# train = os.path.join(dirname, 'train')
# val = os.path.join(dirname, 'val/images')
# test = os.path.join(dirname, 'test/images')
#
# class_ids = os.listdir(train) + os.listdir(val) + os.listdir(test)
#
# print(len(os.listdir(train)), len(glob.glob(train + '/*/images/*.JPEG')),len(os.listdir(val)), len(os.listdir(test)))
# # print(len(set(class_ids)))

# import torch
#
# A = torch.randint(3, (1, 2))
# B = torch.randint(3, (1, 2, 3))
# C = torch.randint(3, (2, 3))
# D = torch.einsum('ij,ijk,jk->ik', A, B, C)
# print('A --- ', A)
# print('B --- ', B)
# print('C --- ', C)
# print(D)


"""
tensorflow-text
"""
import tensorflow as tf
# import tensorflow_text as text
#
# # 设置模型的 UR
# MODEL_HANDLE = "https://hub.tensorflow.google.cn/google/zh_segmentation/1"
# segmenter = text.WhitespaceTokenizer()
# # tokens = tf.strings.unicode_split([u"仅今年前".encode('UTF-8')], 'UTF-8')
#
# # 分割 新华社北京
# input_text = ["I love BeiJing!"]
# tokens, starts, ends = segmenter.tokenize_with_offsets(input_text)
# print(starts, ends)
#
# for i in tokens.to_list()[0]:
#     print(i.decode('utf-8'))
#
#
# loss = tf.nn.softmax_cross_entropy_with_logits(
#     [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
#     [[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]],)
#
#
# print('Loss: ', loss.numpy())  # Loss: 0.0945

indices = [[4, 1, 2],
           [2, 5, 4],
           [2, 3, 1]]

maximum = tf.reduce_max(indices, axis=-1, keepdims=True)
maximum1 = tf.where(indices == maximum, )
print(maximum1)
print(maximum)



















