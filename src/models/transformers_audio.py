import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as  plt
import gc
import load_file as ld
import tensorflow as tf

import itertools
from itertools import cycle

import zipfile

from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import label_ranking_average_precision_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm=np.round(cm, 2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def display_results(Y_test, y_test, pred_probs, cm = True):
    pred = np.argmax(pred_probs, axis=-1)
    print('Test Set Accuracy =  {0:.2f}'.format(accuracy_score(Y_test, pred)))
    print('Test Set F-score =  {0:.2f}'.format(f1_score(Y_test, pred, average='macro')))
    if cm:
        plot_confusion_matrix(confusion_matrix(Y_test, pred), classes=label_dict.keys())

#Acknowledgment: https://github.com/facundodeza/transfomer-audio-classification/blob/master/audio_classification_transformer.ipynb

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)

    return tf.matmul(attention_weights, value)

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    outputs = self.dense(concat_attention)

    return outputs

class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# This allows to the transformer to know where there is real data and where it is padded
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def encoder_layer(units, d_model, num_heads, dropout,name="encoder_layer"):
  inputs = tf.keras.Input(shape=(431,d_model ), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(time_steps,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            projection,
            name="encoder"):
  inputs = tf.keras.Input(shape=(431,d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
  
  if projection=='linear':
    projection=tf.keras.layers.Dense( d_model,use_bias=True, activation='linear')(inputs)
    print('linear')
  
  else:
    projection=tf.identity(inputs)
    print('none')
   
  projection *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  projection = PositionalEncoding(time_steps, d_model)(projection)

  outputs = tf.keras.layers.Dropout(rate=dropout)(projection)

  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])
 
 
  

 
  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def transformer(time_steps,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                output_size,
                projection,
                name="transformer"):
    inputs = tf.keras.Input(shape=(431,d_model), name="inputs")
  
  
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(tf.dtypes.cast(
          
    #Like our input has a dimension of length X d_model but the masking is applied to a vector
    # We get the sum for each row and result is a vector. So, if result is 0 it is because in that position was masked      
    tf.math.reduce_sum(
    inputs,
    axis=2,
    keepdims=False,
    name=None
    ), tf.int32))
  

    enc_outputs = encoder(
        time_steps=time_steps,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        projection=projection,
        name='encoder'
    )(inputs=[inputs, enc_padding_mask])

    #We reshape for feeding our FC in the next step
    outputs=tf.reshape(enc_outputs,(-1,time_steps*d_model))
  
    #We predict our class
    outputs = tf.keras.layers.Dense(units=output_size,use_bias=True,activation='softmax', name="outputs")(outputs)

    return tf.keras.Model(inputs=[inputs], outputs=outputs, name='audio_class')

path=r"C:\Irene\universita\magistrale\Speech_recognition\project"
os.chdir(path)
method = 1 # 1 for MFCC or 2 for MEL
X_train,X_test,Y_train,Y_test=ld.get_data(path, i=method)

label_dict = {'Animal':0,
              'Humans':1,
              'Natural':2,
             }

y_test = np.zeros((len(Y_test),3))

for i in range(len(Y_test)):
    if Y_test[i]==0:
        y_test[i,0] = 1
        y_test[i,1] = 0
        y_test[i,2] = 0
    if Y_test[i]==1:
        y_test[i,0] = 0
        y_test[i,1] = 1
        y_test[i,2] = 0
    if Y_test[i]==2:
        y_test[i,0] = 0
        y_test[i,1] = 0
        y_test[i,2] = 1 

y_train = np.zeros((len(Y_train),3))

for i in range(len(Y_train)):
    if Y_train[i]==0:
        y_train[i,0] = 1
        y_train[i,1] = 0
        y_train[i,2] = 0
    if Y_train[i]==1:
        y_train[i,0] = 0
        y_train[i,1] = 1
        y_train[i,2] = 0
    if Y_train[i]==2:
        y_train[i,0] = 0
        y_train[i,1] = 0
        y_train[i,2] = 1 


n_mfcc=13
test = X_test[1,:]
frame = int(len(test)/n_mfcc)
x_test = np.zeros((len(X_test[:,1]),frame,n_mfcc))
    

for x in range(len(X_test[:,1])):
    for y in range(frame):
        for z in range(n_mfcc):
            x_test[x,y,z] = X_test[x,n_mfcc*y+z]


train = X_train[1,:]
frame = int(len(train)/n_mfcc)
x_train = np.zeros((len(X_train[:,1]),frame,n_mfcc))
    

for x in range(len(X_train[:,1])):
    for y in range(frame):
        for z in range(n_mfcc):
            x_train[x,y,z] = X_train[x,n_mfcc*y+z]

x_train = x_train[:,:,:-1]
x_test = x_test[:,:,:-1]

X = x_train          

#projection=['linear','none']
projection=['linear']
accuracy=[]
proj_implemented=[]

for i in projection:
  NUM_LAYERS = 2
  D_MODEL = X.shape[2]
  NUM_HEADS = 4
  UNITS = 1024
  DROPOUT = 0.1
  TIME_STEPS= X.shape[1]
  OUTPUT_SIZE=3 
  EPOCHS = 10
  EXPERIMENTS= 2

  
  for j in range(EXPERIMENTS):
    
    
    model = transformer(time_steps=TIME_STEPS,
      num_layers=NUM_LAYERS,
      units=UNITS,
      d_model=D_MODEL,
      num_heads=NUM_HEADS,
      dropout=DROPOUT,
      output_size=OUTPUT_SIZE,  
      projection=i  )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.000001), loss='categorical_crossentropy', metrics=['accuracy'])

    
    history=model.fit(x_train,y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    accuracy.append(max(history.history['val_accuracy']))
      
    proj_implemented.append(i)

    pred_probs = model.predict(x_test)
    display_results(Y_test, y_test, pred_probs)
    
    del model
    
    del history

accuracy=pd.DataFrame(accuracy, columns=['accuracy'])
proj_implemented=pd.DataFrame(proj_implemented, columns=['projection'])
results=pd.concat([accuracy,proj_implemented],axis=1)

results.groupby('projection').mean()

results.groupby('projection').std()

print(results)
