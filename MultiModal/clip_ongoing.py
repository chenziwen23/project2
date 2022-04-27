

# 1. data organization

#

from utils import *
from multiprocessing import pool
import os
import collections
import json
import numpy as np
import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
from tensorflow import keras

from tensorflow.keras import layers
import tensorflow_text as tftext
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

# tf.enable_eager_execution()
# tensorflow.python.eager.core._SymbolicException
# tf.compat.v1.disable_eager_execution()      # write to records ： can't use
# tf.config.experimental_run_functions_eagerly(True)
#  MS - coco,  80000+ image, each image contains 5+ caption annotations

root_dir = '/data/COCO2014/'
root_dir_tf = '/students/julyedu_529223/coco2014'

images_dir = root_dir
tfrecords_dir = root_dir_tf
annotations_file = os.path.join(root_dir, 'annotations', 'captions_train2014.json')
annotations_file_val = os.path.join(root_dir, 'annotations', 'captions_val2014.json')

if not os.path.exists(annotations_file):
    annotations_zip = tf.keras.utils.get_file(
        'caption.zip',
        cache_dir = os.path.abspath('.'),
        Ordinal = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        extract = True,

    )
    os.remove(annotations_zip)

if not os.path.exists(images_dir):
    images_zip = tf.keras.utils.get_file(
        'train2014.zip',
        cache_dir = os.path.abspath('.'),
        Ordinal = 'http://images.cocodataset.org/zips/train2014.zip',
        extract = True,

    )
    os.remove(images_zip)

print('Dataset is downloaded and extracted ')

with open(annotations_file, 'r') as f:
    annotations = json.load(f)['annotations']
with open(annotations_file_val, 'r') as f:
    annotations_val = json.load(f)['annotations']

image_path_to_caption = collections.defaultdict(list)
image_path_to_caption_val = collections.defaultdict(list)

for element in annotations:
    caption = f"{element['caption'.lower().rstrip('.')]}"
    image_path = images_dir + 'train2014/COCO_train2014_' + '%012d.jpg' % (element['image_id'])
    image_path_to_caption[image_path].append(caption)

for element in annotations_val:
    caption = f"{element['caption'.lower().rstrip('.')]}"
    image_path = images_dir + 'val2014/COCO_val2014_' + '%012d.jpg' % (element['image_id'])
    image_path_to_caption_val[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
image_paths_val = list(image_path_to_caption_val.keys())

train_size = 20000
valid_size = 2000
# captions_per_image = 3
images_per_file = 2000

train_image_paths = image_paths[:train_size]
num_train_files = int(np.ceil(train_size / images_per_file))
train_files_prefix = os.path.join(tfrecords_dir, 'train')

val_image_paths = image_paths_val[:valid_size]
num_val_files = int(np.ceil(valid_size / images_per_file))
val_files_prefix = os.path.join(tfrecords_dir, 'valid')

tf.io.gfile.makedirs(tfrecords_dir)


def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))


def create_example(image_path, caption):
    feature = {
        'caption': bytes_feature(caption.encode()),
        'raw_image': bytes_feature(tf.io.read_file(image_path).numpy())
        }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(file_name, image_paths):
    caption_list = []
    image_path_list = []
    for image_path in image_paths:
        caption = image_path_to_caption_val[image_path]
        caption_list.extend(caption)
        image_path_list.extend([image_path] * len(caption))

    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx in range(len(image_path_list)):
            example = create_example(image_path_list[example_idx], caption_list[example_idx])
            writer.write(example.SerializeToString())
    return example_idx + 1


def write_data(image_path, num_file, files_prefix):
    example_counter = 0
    for file_idx in tqdm(range(num_file)):
        file_name = files_prefix + '-%02d.tfrecords'%(file_idx)
        start_idx = images_per_file * file_idx
        end_idx = start_idx + images_per_file

        example_counter += write_tfrecords(file_name, image_path[start_idx:end_idx])
    return example_counter


# train_example_counter = write_data(train_image_paths, num_train_files, train_files_prefix)
# print(f"{train_example_counter} training examples were wrritten to tfrecord files.")
#
# valid_example_counter = write_data(val_image_paths, num_val_files, val_files_prefix)
# print(f"{valid_example_counter} evaluation examples were wrritten to tfrecord files.")


feature_description = {
    'caption': tf.io.FixedLenFeature([], tf.string),
    'raw_image': tf.io.FixedLenFeature([], tf.string)
}


def read_example(example):
    features = tf.io.parse_single_example(example, feature_description)
    raw_images = features.pop('raw_image')
    features['images'] = tf.image.resize(
        tf.image.decode_jpeg(raw_images, channels=3), size=(299, 299)
    )
    return features


def get_datasets(file_pattern, batch_size):
    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern)).map(
            read_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            # daterministic = False,
        )
        .shuffle(batch_size * 10)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
    )

# Model Arichetecture

def project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate):
    projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings


# Vision Encoder
def create_vision_encoder(num_projecton_layers, project_dims, dropout_rate, trainable=False):
    # load the pretained model (Xception) to be used as the base encoder

    xception = keras.applications.Xception(
        include_top = False, weights='imagenet', pooling = 'avg'
    )

    # set trainability of the base encoder
    for layer in xception.layers:
        layer.trainable = trainable

    # feedforward and get the embeddings
    inputs = layers.Input(shape = (299, 299, 3), name = 'image_input')
    xception_input = keras.applications.xception.preprocess_input(inputs)
    embeddings = xception(xception_input)

    # project the embeddings produced by the model
    outputs = project_embeddings(embeddings, num_projecton_layers, project_dims, dropout_rate)

    return keras.Model(inputs=inputs,  outputs=outputs, name = 'vision_encoder')


# Text Encoder
# RNN / LSTM / Word Embedding (Word2Vec) / Bert
def create_text_encoder(num_projecton_layers, project_dims, dropout_rate, trainable=False):
    # load the BERT preprocessing model
    preprocess = hub.KerasLayer(
        "/students/julyedu_529223/project/MultiModal/bert_en_uncased_preprocess"
    )
    bert = hub.KerasLayer(
        "/students/julyedu_529223/project/MultiModal/small_bert_bert",
        "bert"
    )

    # set the trainability of the base encoder
    bert.trainable = trainable

    inputs = layers.Input(shape=(), dtype=tf.string, name="inputs")

    # Preprocess the text first
    bert_inputs = preprocess(inputs)

    # Get the embeddings for the preprocessed text using the BERT model
    embeddings = bert(bert_inputs)['pooled_output']

    # project the embeddings produced by the model
    projected_embeddings = project_embeddings(embeddings, num_projecton_layers, project_dims, dropout_rate)

    return keras.Model(inputs=inputs,  outputs=projected_embeddings, name='text_encoder')


Text_encoder = create_text_encoder(1, 256, 0.5, trainable=True)
Image_encoder = create_vision_encoder(1, 256, 0.5, trainable=True)
Text_encoder.summary()
# load model: you will encounter an error prompt if the model highly customized
# load_model .  inference

# batch data: [caption 1, caption 2, ...., caption i, ...],  [image 1, image 2, ...., image i, ...]
# To compute the loss
class DualEncoder(keras.Model):
    def __init__(self, text_encoder, image_encoder, temperature=1.0, **kwargs):
        # super(DualEncoder, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False, **kwargs):
        # we could place each encoder on a separate GPU if available
        # Tensorflow will fallback on the available devices if there are fewer than 2 gpus
        with tf.device('/gpu:0'):
            # Get the embeddings for the caption
            caption_embedding = Text_encoder(features['caption'], training=training)   # , trainable=training
        with tf.device('/gpu:1'):
            # Get the embeddings for the caption
            image_embedding = Image_encoder(features['images'], training=training)      # , trainable=training

        return caption_embedding, image_embedding

    def compute_loss(self, caption_embeddings, image_embeddings):
        # logits[i][j] is the dot_similarity(caption_i, caption_j)
        logits = tf.matmul(caption_embeddings, image_embeddings, transpose_b=True) * tf.math.exp(self.temperature)

        # images_similarity[i][j] is the dot_similarity(image_i, image_j)
        images_similarity = tf.matmul(image_embeddings, image_embeddings, transpose_b=True)
        captions_similarity = tf.matmul(caption_embeddings, caption_embeddings, transpose_b=True)
        # maximum = tf.reduce_max((images_similarity + captions_similarity), axis=-1, keepdims=True)
        maximum = (images_similarity + captions_similarity)
        targets = keras.activations.softmax(maximum)

        loss_caption = keras.losses.categorical_crossentropy(
            targets, logits, from_logits=True
        )

        loss_image = keras.losses.categorical_crossentropy(
            tf.transpose(targets), tf.transpose(logits), from_logits = True
        )

        final_loss = (loss_image + loss_caption) / 2.

        # Return the mean of the loss over the batch.
        return final_loss

    def train_step(self, features):
        with tf.GradientTape() as tape:
            # forward
            caption_embeddings, images_embeddings = self(features, training=True)
            loss = self.compute_loss(caption_embeddings, images_embeddings)

        # Backward
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Monitor
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        caption_embeddings, images_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, images_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# # Train the dual encoder model
# num_epoch = 20
# batch_size = 256
# print("******************************************************************************************************")
# print("********************************  Create DualEncoder Module ......... ********************************")
# print("******************************************************************************************************")
# dual_encoder = DualEncoder(create_text_encoder, create_vision_encoder, temperature=0.05)
#
# dual_encoder.compile(
#     optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001),
# )
#
# # v100  , batch_size = 256, dual time: 10-12mins
# # tf.config.list_physical_devices('GPU')
#
# train_dataset = get_datasets(os.path.join(tfrecords_dir, "train-*.tfrecords"), batch_size)
# valid_dataset = get_datasets(os.path.join(tfrecords_dir, "valid-*.tfrecords"), batch_size)
#
# record_lr = keras.callbacks.ReduceLROnPlateau(monitor= 'val_loss', factor=0.2, patience=3)
#
# early_stopping = keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=5, restore_best_weights=True)
#
# history = dual_encoder.fit(
#     train_dataset,
#     epochs=num_epoch,
#     validation_data=valid_dataset,
#     callbacks=[record_lr, early_stopping]
# )
#
# print("Training completed. Saving vision and text encoders .... ")
# Image_encoder.save("image_encoder")
# Text_encoder.save("text_encoder")
#
# print("Model are saved ....")


# inference / visualization / evaluation
batch_size = 256
Image_encoder = keras.models.load_model("image_encoder")
Text_encoder = keras.models.load_model("text_encoder")


def read_image(image_path):
    image_array = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return tf.image.resize(image_array, (299, 299))

image_embeddings = Image_encoder.predict(
    tf.data.Dataset.from_tensor_slices(image_paths).map(read_image).batch(batch_size),
    verbose=1
)

print(f"Image Embeddings shape: {image_embeddings.shape}")


# retrieve relevant images demo


# In our project, we will use extract matching (linear scan / brute force) by computing the dot similarity between
# the input query-embedding and the image (database) embeddings .
# Then it will retrieve the Top-k matching results. If you prefer in real-time use cases or large scale data,
# approximate nearest neighbour(ANN), e.g. random hashing such as [Faiss] is what you need.

def find_matching(image_embeddings, queries, k = 9, normalize = True):
    # Get the
    # query is often a description
    query_embedding = Text_encoder(tf.convert_to_tensor(queries))

    # Normalize the query and the image embeddings
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)

    # Compute the dot product between the query and the image embeddings
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)

    # Retrieve top K indices
    results = tf.math.top_k(dot_similarity, k=k).indices.numpy()

    return [[image_paths[idx] for idx in indices] for indices in results]


def visualization():
    query = "a handsome boy walking his dog."
    matches = find_matching(image_embeddings, [query], normalize=True)[0]

    plt.figure(figsize=(20, 20))

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(mpimg.imread(matches[i]))
        ax.axis('off')
    plt.axis('off')
    plt.savefig('haha.jpg')
    plt.show()

# Evaluate the retrieval quality

# In order to evaluate the dual encoder model, we will use the captions as queries
# Specifically, we will use the out-of-training-sample images and captions to evaluate the retrieval quality,
# by using top-k accuracy. A true prediction is counted if, for a given caption, its associated image is retrieved
# within the top k matches.

def compute_top_k_accuracy(image_paths, image_path2caption, k=9):
    hits = 0
    num_batches = int(np.ceil(len(image_paths) / batch_size))

    for idx in tqdm(range(num_batches)):
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size
        current_image_paths = image_paths[start_idx:end_idx]

        queries = [image_path2caption[image_path][0] for image_path in current_image_paths]

        result = find_matching(image_embeddings, queries, k)

        hits += sum(
            [image_path in matches for (image_path, matches) in list(zip(current_image_paths, result))]
        )

    return hits / len(image_paths)


def evaluation():
    # training data
    training_accuracy = compute_top_k_accuracy(train_image_paths, image_path_to_caption)

    # evaluating data
    evaluation_accuracy = compute_top_k_accuracy(image_paths_val[valid_size:], image_path_to_caption_val)
    print(f"------ training_accuracy:  {training_accuracy * 100} %   "
          f"\n------ evaluation_accuracy:  {evaluation_accuracy * 100} %")


visualization()
evaluation()
