# Import dependencies
import os
from collections import Counter
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import metrics
import keras
import keras_tuner

# Define path to corpus
corpus_path = r"gc_training_data\en"

# Load dataset from files
train = []
with open(os.path.join(corpus_path, r'train.txt'), 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines[1:]:
      sent = line.split("\t")[0]
      label = line.split("\t")[1]
      if label == '0\n':
        train.append([sent, 0])
      elif label == '1\n':
        train.append([sent, 1])

val = []
with open(os.path.join(corpus_path, r'val.txt'), 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines[1:]:
      sent = line.split("\t")[0]
      label = line.split("\t")[1]
      if label == '0\n':
        train.append([sent, 0])
      elif label == '1\n':
        train.append([sent, 1])

test = []
with open(os.path.join(corpus_path, r'test.txt'), 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines[1:]:
      sent = line.split("\t")[0]
      label = line.split("\t")[1]
      if label == '0\n':
        train.append([sent, 0])
      elif label == '1\n':
        train.append([sent, 1])

lists = [train, val, test]
lens = [len(li) for li in lists]

print('Loaded dataset with {} sentences.'.format(sum(lens)))
print('Sentences in train: ', len(train))
print('Sentences in validation: ', len(val))
print('Sentences in test: ', len(test))

# Collect expected labels
y_train = np.array([sample[1] for sample in train])
y_val = np.array([sample[1] for sample in val])
y_test = np.array([sample[1] for sample in test])

lists = [y_train, y_val, y_test]
lens = [len(li) for li in lists]

print('{} labels collected'.format(sum(lens)))

print('Label distribution in training set: ', Counter(Counter(y_train)))
print('Label distribution in validation set: ', Counter(Counter(y_val)))
print('Label distribution in test set: ', Counter(Counter(y_test)))

# Compute statistics on dataset
dataset = train + val + test

def tokenize(dataset):
  """
  Tokenize dataset
  from samples in format [[sent, label]]
  Necessary for calculating TF-IDF
  """
  tokenized_dataset = []
  for sample in dataset:
    tokenized_sent = word_tokenize(sample[0], language='english')
    tokenized_dataset.append(tokenized_sent)
  return tokenized_dataset

tokenized_dataset = tokenize(dataset)

print('Longest sentence in dataset:', len(max(tokenized_dataset, key=len)))

vocab = []
for sent in tokenized_dataset:
  for tok in sent:
    vocab.append(tok)
types = set(vocab)

print('Vocabulary: {} words'.format(len(vocab)))
print('Unique words:', len(types))

# Collect sentences
x_train = [sample[0].lower() for sample in train]
x_val = [sample[0].lower() for sample in val]
x_test = [sample[0].lower() for sample in test]

## Build models
# CNN
class CNN_HyperModel(keras_tuner.HyperModel):
  def __init__(self, maxlen, embedding_dims):
    self.maxlen = maxlen
    self.embedding_dims = embedding_dims

  def build(self, hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(self.maxlen, self.embedding_dims)))
    model.add(keras.layers.Conv1D(filters = hp.Int("filters", min_value=100, max_value=500, step=50),
                     kernel_size = hp.Int("kernel_size", min_value=3, max_value=7, step=1),
                     padding='same',
                     activation='relu',
                     strides=1))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(self.embedding_dims))
    model.add(keras.layers.Dropout(hp.Choice('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(
      learning_rate=hp.Choice('learning_rate', [0.01, 0.005, 0.001, 0.0005, 0.0001])),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'],
                  run_eagerly=True)
    return model

  def fit(self, hp, model, *args, **kwargs):
    return model.fit(
        *args,
        batch_size=hp.Choice('batch_size', [32, 64, 128]),
        **kwargs)

# LSTM
class LSTM_HyperModel(keras_tuner.HyperModel):
  def __init__(self, maxlen, embedding_dims):
    self.maxlen = maxlen
    self.embedding_dims = embedding_dims

  def build(self, hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(self.maxlen, self.embedding_dims)))
    model.add(keras.layers.LSTM(units = hp.Int('units', min_value=50, max_value=250, step=10)))
    model.add(keras.layers.Dropout(hp.Choice('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=keras.optimizers.RMSprop(
      learning_rate=hp.Choice('learning_rate', [0.01, 0.005, 0.001, 0.0005, 0.0001])),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model

  def fit(self, hp, model, *args, **kwargs):
    return model.fit(*args,
                     batch_size=hp.Choice('batch_size', [32, 64, 128]),
                     **kwargs)


### FIT MODELS ###

### 1. Word2Vec

# Load pre-trained vectors
from gensim.models.keyedvectors import KeyedVectors
w2v_vectors = KeyedVectors.load_word2vec_format(
    r"embeddings\GoogleNews-vectors-negative300.bin.gz",
    binary=True)

# Define function to vectorize sentences
def embed(dataset, word_embeddings, language):
    vectorized_data = []
    for sample in dataset:
        tokens = word_tokenize(sample[0].lower(), language=language)
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_embeddings[token])
            except KeyError:
                pass # no matching token in the w2v vocab: discard
        vectorized_data.append(sample_vecs)
    
    return vectorized_data

w2v_train = embed(train, w2v_vectors, 'english')
w2v_val = embed(val, w2v_vectors, 'english')
w2v_test = embed(test, w2v_vectors, 'english')

w2v_dataset = w2v_train + w2v_val + w2v_test

# Pad/truncate vectorized dataset
maxlen = len(max(w2v_dataset, key=len)) # length of longest vector
print('W2V vector length: ', maxlen)

def pad_trunc(data, maxlen):
    """
    Pad or truncate samples in dataset based on maxlen (input length)
    and vectorize new data
    """
    new_data = []
    # Create zero-vector the length of word vectors
    zero_vector = []
    for i in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            # Append the appropriate number of zero-vectors to the list
            for i in range(maxlen - len(sample)):
                temp.append(zero_vector)
        else:
            temp = sample

        new_data.append(temp)
    
    vectorized_data = np.array(new_data)
    return vectorized_data

w2v_xtrain = pad_trunc(w2v_train, maxlen)
w2v_xval = pad_trunc(w2v_val, maxlen)
w2v_xtest = pad_trunc(w2v_test, maxlen)

w2v_embedding_dims = len(w2v_dataset[0][0]) # word embedding dimensions

## 1.1. CNN
# Instantiate tuner
cnn_w2v_tuner = keras_tuner.Hyperband(
    hypermodel = CNN_HyperModel(maxlen=maxlen, embedding_dims=w2v_embedding_dims),
    objective="val_loss",
    max_epochs=15,
    seed=42,
    overwrite=True,
    directory=r"keras_tuners",
    project_name="cnn_w2v_en"
)

# Search best hyperparameters
cnn_w2v_tuner.search(w2v_xtrain, y_train,
                     callbacks=[
                        keras.callbacks.EarlyStopping("val_loss", patience=1, restore_best_weights=True),
                        keras.callbacks.TensorBoard(log_dir=r"tensorboard/cnn_w2v_en")
                     ],
                     validation_data=(w2v_xval,  y_val))

# Retrieve best hyperparameters
best_hps = cnn_w2v_tuner.get_best_hyperparameters()[0]
print(best_hps.values)

# Retrieve best model
cnn_w2v = cnn_w2v_tuner.get_best_models()[0]
cnn_w2v.summary()

## Best model evaluation
# Get predictions
predictions = cnn_w2v.predict(w2v_xtest)
# Binarize
y_pred = []
for pred in predictions:
  y_pred.append((pred > 0.5).astype(int))
# Accuracy
print('Accuracy score for CNN with W2V:', metrics.accuracy_score(y_test, y_pred), '\n')
# Main classification metrics
print('Classification report for CNN with W2V')
print(metrics.classification_report(y_test, y_pred, zero_division=0.0), '\n')
# Confusion matrix
print('Confusion matrix for CNN with W2V')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)))

# Export model
cnn_w2v.save(r"models\cnn_w2v_en.keras")
cnn_w2v.save_weights(r"models\cnn_w2v_en.weights.h5")

## 1.2. LSTM
# Instantiate tuner
lstm_w2v_tuner = keras_tuner.Hyperband(
    hypermodel = LSTM_HyperModel(maxlen=maxlen, embedding_dims=w2v_embedding_dims),
    objective="val_loss",
    max_epochs=15,
    seed=42,
    overwrite=True,
    directory=r"keras_tuners",
    project_name=r"lstm_w2v_en"
)

# Search best hyperparameters
lstm_w2v_tuner.search(w2v_xtrain, y_train,
                      callbacks=[
                         keras.callbacks.EarlyStopping("val_loss", patience=2, restore_best_weights=True),
                         keras.callbacks.TensorBoard(log_dir=r"tensorboard/lstm_w2v_en")
                      ],
                      validation_data=(w2v_xval,  y_val))

# Retrieve best hyperparameters
best_hps = lstm_w2v_tuner.get_best_hyperparameters()[0]
print(best_hps.values)

# Retrieve best model
lstm_w2v = lstm_w2v_tuner.get_best_models()[0]
lstm_w2v.summary()

## Best model evaluation
# Get predictions
predictions = lstm_w2v.predict(w2v_xtest)
# Binarize
y_pred = []
for pred in predictions:
  y_pred.append((pred > 0.5).astype(int))
# Accuracy
print('Accuracy score for LSTM with W2V:', metrics.accuracy_score(y_test, y_pred), '\n')
# Main classification metrics
print('Classification report for LSTM with W2V')
print(metrics.classification_report(y_test, y_pred, zero_division=0.0), '\n')
# Confusion matrix
print('Confusion matrix for LSTM with W2V')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)))

# Export model
lstm_w2v.save(r"models\lstm_w2v_en.keras")
lstm_w2v.save_weights(r"models\lstm_w2v_en.weights.h5")


### 2. fastText
# Import pre-trained vectors
from gensim.models.fasttext import load_facebook_vectors
ft_vectors = load_facebook_vectors(r"..\embeddings\cc.en.300.bin.gz",
                                   encoding='utf-8')

ft_train = embed(train, ft_vectors, 'english')
ft_val = embed(val, ft_vectors, 'english')
ft_test = embed(test, ft_vectors, 'english')

ft_dataset = ft_train + ft_val + ft_test

# Pad/truncate dataset
maxlen = len(max(ft_dataset, key=len))

ft_xtrain = pad_trunc(ft_train, maxlen)
ft_xval = pad_trunc(ft_val, maxlen)
ft_xtest = pad_trunc(ft_test, maxlen)

ft_embedding_dims = len(ft_dataset[0][0])

## 2.1. CNN
# Instantiate tuner
cnn_ft_tuner = keras_tuner.Hyperband(
    hypermodel = CNN_HyperModel(maxlen=maxlen, embedding_dims=ft_embedding_dims),
    objective="val_loss",
    max_epochs=15,
    seed=42,
    overwrite=True,
    directory=r"keras_tuners",
    project_name=r"cnn_ft_en",
)

# Search best hyperparameters
cnn_ft_tuner.search(ft_xtrain, y_train,
                    callbacks=[
                       keras.callbacks.EarlyStopping("val_loss", patience=1, restore_best_weights=True),
                       keras.callbacks.TensorBoard(log_dir=r"tensorboard\cnn_ft_en")
                    ],
                    validation_data=(ft_xval,  y_val))

# Retrieve best hyperparameters
best_hps = cnn_ft_tuner.get_best_hyperparameters()[0]
print(best_hps.values)

# Retrieve best model
cnn_ft = cnn_ft_tuner.get_best_models()[0]
cnn_ft.summary()

## Best model evaluation
# Get predictions
predictions = cnn_ft.predict(ft_xtest)
# Binarize
y_pred = []
for pred in predictions:
  y_pred.append((pred > 0.5).astype(int))
# Accuracy
print('Accuracy score for CNN with FT:', metrics.accuracy_score(y_test, y_pred), '\n')
# Main classification metrics
print('Classification report for CNN with FT')
print(metrics.classification_report(y_test, y_pred, zero_division=0.0), '\n')
# Confusion matrix
print('Confusion matrix for CNN with FT')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)))

# Export model
cnn_ft.save(r"models\cnn_ft_en.keras")
cnn_ft.save_weights(r"models\cnn_ft_en.weights.h5")

## 2.2. LSTM
# Instantiate tuner
lstm_ft_tuner = keras_tuner.Hyperband(
    hypermodel = LSTM_HyperModel(maxlen=maxlen, embedding_dims=ft_embedding_dims),
    objective="val_loss",
    max_epochs=15,
    seed=42,
    overwrite=True,
    directory=r"keras_tuners",
    project_name=r"lstm_ft_en",
)

# Search best hyperparameters
lstm_ft_tuner.search(ft_xtrain, y_train,
                     callbacks=[
                        keras.callbacks.EarlyStopping("val_loss", patience=2, restore_best_weights=True),
                        keras.callbacks.TensorBoard(log_dir=r"tensorboard/lstm_ft")
                     ],
                     validation_data=(ft_xval,  y_val))

# Retrieve best hyperparameters
best_hps = lstm_ft_tuner.get_best_hyperparameters()[0]
print(best_hps.values)

# Retrieve best model
lstm_ft = lstm_ft_tuner.get_best_models()[0]
lstm_ft.summary()

## Best model evaluation
# Get predictions
predictions = lstm_ft.predict(ft_xtest)
# Binarize
y_pred = []
for pred in predictions:
  y_pred.append((pred > 0.5).astype(int))
# Accuracy
print('Accuracy score for LSTM with FT:', metrics.accuracy_score(y_test, y_pred), '\n')
# Main classification metrics
print('Classification report for LSTM with FT')
print(metrics.classification_report(y_test, y_pred, zero_division=0.0), '\n')
# Confusion matrix
print('Confusion matrix for LSTM with FT')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)))

# Export model
lstm_ft.save(r"models\lstm_ft_en.keras")
lstm_ft.save_weights(r"models\lstm_ft_en.weights.h5")


### 3. GloVe
import gensim.downloader as api
glove_vectors = api.load("glove-wiki-gigaword-300")

glove_train = embed(train, glove_vectors, 'english')
glove_val = embed(val, glove_vectors, 'english')
glove_test = embed(test, glove_vectors, 'english')

glove_dataset = glove_train + glove_val + glove_test

# Pad/truncate dataset
maxlen = len(max(glove_dataset, key=len)) # length of longest vector
print('GloVe vector length: ', maxlen)

glove_xtrain = pad_trunc(glove_train, maxlen)
glove_xval = pad_trunc(glove_val, maxlen)
glove_xtest = pad_trunc(glove_test, maxlen)

glove_embedding_dims = len(glove_dataset[0][0]) # word embedding dimensions
print('GloVe embedding dimensions: ', glove_embedding_dims)

## 3.1. CNN
# Instantiate tuner
cnn_glove_tuner = keras_tuner.Hyperband(
    hypermodel = CNN_HyperModel(maxlen=maxlen, embedding_dims=glove_embedding_dims),
    objective="val_loss",
    max_epochs=15,
    seed=42,
    overwrite=True,
    directory=r"keras_tuners",
    project_name=r"cnn_glove_en",
)

# Search best hyperparameters
cnn_glove_tuner.search(glove_xtrain, y_train,
                    callbacks=[
                       keras.callbacks.EarlyStopping("val_loss", patience=1, restore_best_weights=True),
                       keras.callbacks.TensorBoard(log_dir=r"tensorboard/cnn_glove_en")
                    ],
                    validation_data=(glove_xval,  y_val))

# Retrieve best hyperparameters
best_hps = cnn_glove_tuner.get_best_hyperparameters()[0]
print(best_hps.values)

# Retrieve best model
cnn_glove = cnn_glove_tuner.get_best_models()[0]
cnn_glove.summary()

## Best model evaluation
# Get predictions
predictions = cnn_glove.predict(glove_xtest)
# Binarize
y_pred = []
for pred in predictions:
  y_pred.append((pred > 0.5).astype(int))
# Accuracy
print('Accuracy score for CNN with GloVE:', metrics.accuracy_score(y_test, y_pred), '\n')
# Main classification metrics
print('Classification report for CNN with GloVe')
print(metrics.classification_report(y_test, y_pred, zero_division=0.0), '\n')
# Confusion matrix
print('Confusion matrix for CNN with GloVe')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)))

# Export model
cnn_glove.save(r"models\cnn_glove_en.keras")
cnn_glove.save_weights(r"models\cnn_glove_en.weights.h5")

## 3.2. LSTM
# Instantiate tuner
lstm_glove_tuner = keras_tuner.Hyperband(
    hypermodel = LSTM_HyperModel(maxlen=maxlen, embedding_dims=glove_embedding_dims),
    objective="val_loss",
    max_epochs=15,
    seed=42,
    overwrite=True,
    directory=r"keras_tuners",
    project_name=r"lstm_glove_en",
)

# Search best hyperparameters
lstm_glove_tuner.search(glove_xtrain, y_train,
                     callbacks=[
                        keras.callbacks.EarlyStopping("val_loss", patience=2, restore_best_weights=True),
                        keras.callbacks.TensorBoard(log_dir=r"tensorboard\lstm_glove")
                     ],
                     validation_data=(glove_xval,  y_val))

# Retrieve best hyperparameters
best_hps = lstm_glove_tuner.get_best_hyperparameters()[0]
print(best_hps.values)

# Retrieve best model
lstm_glove = lstm_glove_tuner.get_best_models()[0]
lstm_glove.summary()

## Best model evaluation
# Get predictions
predictions = lstm_glove.predict(glove_xtest)
# Binarize
y_pred = []
for pred in predictions:
  y_pred.append((pred > 0.5).astype(int))
# Accuracy
print('Accuracy score for LSTM with FT:', metrics.accuracy_score(y_test, y_pred), '\n')
# Main classification metrics
print('Classification report for LSTM with FT')
print(metrics.classification_report(y_test, y_pred, zero_division=0.0), '\n')
# Confusion matrix
print('Confusion matrix for LSTM with FT')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)))

# Export model
lstm_glove.save(r"models\lstm_glove_en.keras")
lstm_glove.save_weights(r"models\lstm_glove_en.weights.h5")


### 4. RoBERTa
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

# Tokenize each partition (sentences only)
# using RoBERTa
# with maximum sequence length set at 100
roberta_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

roberta_train_tokenized = roberta_tokenizer(x_train, padding=True, truncation=True,
                                            max_length = 100, return_tensors='tf')
roberta_val_tokenized = roberta_tokenizer(x_val, padding=True, truncation=True,
                                            max_length = 100, return_tensors='tf')
roberta_test_tokenized = roberta_tokenizer(x_test, padding=True, truncation=True,
                                            max_length = 100, return_tensors='tf')

# Generate embeddings in batches, then concatenate
# for Keras models
def generate_embeddings(model, tokenized_data, batch_size):
    embeddings_list = []
    for i in range(0, len(tokenized_data['input_ids']), batch_size):
        inputs = {key: value[i:i+batch_size] for key, value in tokenized_data.items()}
        outputs = model(inputs)
        embeddings_list.append(outputs.last_hidden_state)
    return(embeddings_list)

roberta_model = TFAutoModel.from_pretrained("xlm-roberta-large")

batch_size = 64

train_embeddings = generate_embeddings(roberta_model, roberta_train_tokenized, batch_size)
val_embeddings = generate_embeddings(roberta_model, roberta_val_tokenized, batch_size)
test_embeddings = generate_embeddings(roberta_model, roberta_test_tokenized, batch_size)

roberta_x_train = tf.concat(train_embeddings, axis=0)
print('RoBERTa train tensor shape: ', roberta_x_train.shape)
roberta_x_val = tf.concat(val_embeddings, axis=0)
print('RoBERTa validation tensor shape: ', roberta_x_val.shape)
roberta_x_test = tf.concat(test_embeddings, axis=0)
print('Test tensor shape: ', roberta_x_test.shape)

# Define maxlen and embedding dimensions
maxlen = roberta_x_train.shape[1]
print('Max sequence length: ', maxlen)
roberta_embedding_dims = roberta_x_train.shape[2]
print('RoBERTa word embedding dimensions: ', roberta_embedding_dims)

## 4.1. CNN
# Instantiate tuner
cnn_roberta_en_tuner = keras_tuner.Hyperband(
    hypermodel = CNN_HyperModel(maxlen=maxlen, embedding_dims=roberta_embedding_dims),
    objective="val_loss",
    max_epochs=15,
    seed=42,
    overwrite=True,
    directory=r"keras_tuners",
    project_name=r"cnn_roberta_en"
)

# Search best hyperparameters
cnn_roberta_en_tuner.search(roberta_x_train, y_train,
                     callbacks=[
                        keras.callbacks.EarlyStopping("val_loss", patience=1, restore_best_weights=True),
                        keras.callbacks.TensorBoard(log_dir=r"tensoboard\cnn_roberta_en")
                     ],
                     validation_data=(roberta_x_val, y_val))

# Retrieve best hyperparameters
best_hps = cnn_roberta_en_tuner.get_best_hyperparameters()[0]
print(best_hps.values)

# Retrieve best model
cnn_roberta = cnn_roberta_en_tuner.get_best_models()[0]
cnn_roberta.summary()

## Evaluate best model
# Get predictions
predictions = cnn_roberta.predict(roberta_x_test)
# Binarize
y_pred = []
for pred in predictions:
  y_pred.append((pred > 0.5).astype(int))
# Accuracy
print('Accuracy score for CNN with RoBERTa embeddings:', metrics.accuracy_score(y_test, y_pred), '\n')
# Main classification metrics
print('Classification report for CNN with RoBERTa embeddings')
print(metrics.classification_report(y_test, y_pred, zero_division=0.0), '\n')
# Confusion matrix
print('Confusion matrix for CNN with RoBERTa embeddings')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred)))

## Export re-trained model
cnn_roberta.save(r"models\cnn_roberta_en.keras")
cnn_roberta.save_weights(r"models\cnn_roberta_en.weights.h5")

## 4.2. LSTM
## Tune model hyperparameters
# Instantiate tuner
lstm_roberta_tuner = keras_tuner.Hyperband(
    hypermodel = LSTM_HyperModel(maxlen=maxlen, embedding_dims=roberta_embedding_dims),
    objective="val_loss",
    max_epochs=15,
    seed=42,
    overwrite=True,
    directory=r"keras_tuners",
    project_name=r"lstm_roberta_en"
)

# Search best hyperparameters
lstm_roberta_tuner.search(roberta_x_train, y_train,
                     callbacks=[
                        keras.callbacks.EarlyStopping("val_loss", patience=1, restore_best_weights=True),
                        keras.callbacks.TensorBoard(log_dir=r"tensorboard\cnn_roberta_en")
                     ],
                     validation_data=(roberta_x_val, y_val))

# Retrieve best hyperparameters
best_hps = lstm_roberta_tuner.get_best_hyperparameters()[0]
print(best_hps.values)

# Retrieve best model
lstm_roberta = lstm_roberta_tuner.get_best_models()[0]
lstm_roberta.summary()

## Export model
lstm_roberta.save(r"models\lstm_roberta_en.keras")
lstm_roberta.save_weights(r"models\lstm_roberta_en.weights.h5")