# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
from unittest import result
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import pandas as pd
from scipy.interpolate import make_interp_spline
from sklearn.cluster import DBSCAN



from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras import backend as tf_keras_backend

from autocorrect import spell

def recognize(image):
  pass

def newpage2word(path):
  image = cv2.imread(path, 0) # gray scale

  # thresholding
  histr = cv2.calcHist([image],[0],None,[256],[0,256])
  hist = histr.flatten()
  #hist = hist*-1
  peaks, _ = find_peaks(hist, distance=100)

  print(peaks)
  peak_break = peaks[0] + 100

  # getting the thresholded image
  ret,img = cv2.threshold(image,peak_break,255,cv2.THRESH_BINARY)
  img = image

  # histogram and smoothing
  row_hist = np.sum(img, axis=1).tolist()
  row_histn = np.array(row_hist)
  x= np.arange(0,len(row_histn),1)
  smooo = 100
  ratio_changed = len(row_histn)/smooo
  X_ = np.linspace(0, len(row_histn), smooo)
  smooth_row = make_interp_spline(x, row_histn)
  Y_ = smooth_row(X_)
  peaks, _ = find_peaks(Y_)
  actual_peaks = np.array(peaks*ratio_changed, dtype=int)

  # median shit
  test = np.ediff1d(actual_peaks)
  median_linespace = np.median(test)
  mean_linespace = np.mean(test)

  new_lines = []
  item = actual_peaks[0]
  new_lines.append(item)
  for line in range(len(actual_peaks)):
      if line == 0:
        continue
      else: 
        if item + median_linespace/1.5 < actual_peaks[line]:
          new_lines.append(actual_peaks[line])
          item = actual_peaks[line]

  
  final_line_images = []
  start = 0
  for i in new_lines:
    final_line_images.append(img[start:i, :])
    start = i


  # for each line 
  final_word_images = []
  for line_image in final_line_images:
    
    col_hist = np.sum(line_image, axis=0).tolist()
    maximum = np.max(col_hist)
    diff = np.diff(col_hist)
    gradient = np.sign(diff)
    locations = np.where(gradient == 0)[0]
    locations = np.where(col_hist>0.99*maximum)[0]
    
    newloc = locations  

    # clustering: dbscan
    width_of_image = len(line_image[0])
    lasson = pd.DataFrame(newloc, columns=['pizza'])

    dbscan=DBSCAN(eps=5, min_samples=5)
    dbscan.fit(lasson)

    lasson['labels'] = dbscan.labels_
    lasson_drop = lasson[lasson['labels']!= -1]
    lasson_drop.reset_index(inplace = True)

    new_lasson = np.array(lasson_drop.groupby(lasson_drop.labels)[['pizza']].median()['pizza'], dtype='int')


    # cropping page to line
    
    start = new_lasson[0]
    for i in new_lasson[1:]:
      final_word_images.append(line_image[:, start:int(i)])
      start = int(i)

    pav = len(line_image[0])-1



    final_word_images.append(line_image[:, start:pav])
    # PLOTTIGN ALL CROPPED IMAGES
    n = len(final_word_images)

    #print('len of this page', final_word_images[0].shape)
    count = 0
    print('\n\n\nthis')
    print(len(final_word_images))
    print('\n\n\n')
    # for i in final_word_images:
    #   count += 1
    #   plt.subplot(n,1,count)
    #   plt.imshow(i, cmap='gray', vmin=0, vmax=255)

    # plt.show()


  return final_word_images

def page2word(path):
  image = cv2.imread(path, 0) # gray scale

  # thresholding
  histr = cv2.calcHist([image],[0],None,[256],[0,256])
  hist = histr.flatten()
  #hist = hist*-1
  peaks, _ = find_peaks(hist, distance=100)

  print(peaks)
  peak_break = peaks[0] + 100

  # getting the thresholded image
  ret,img = cv2.threshold(image,peak_break,255,cv2.THRESH_BINARY)
  img = image

  # histogram and smoothing
  row_hist = np.sum(img, axis=1).tolist()
  row_histn = np.array(row_hist)
  x= np.arange(0,len(row_histn),1)
  smooo = 100
  ratio_changed = len(row_histn)/smooo
  X_ = np.linspace(0, len(row_histn), smooo)
  smooth_row = make_interp_spline(x, row_histn)
  Y_ = smooth_row(X_)
  peaks, _ = find_peaks(Y_)
  actual_peaks = np.array(peaks*ratio_changed, dtype=int)

  # median shit
  test = np.ediff1d(actual_peaks)
  median_linespace = np.median(test)
  mean_linespace = np.mean(test)

  new_lines = []
  item = actual_peaks[0]
  new_lines.append(item)
  for line in range(len(actual_peaks)):
      if line == 0:
        continue
      else: 
        if item + median_linespace/1.5 < actual_peaks[line]:
          new_lines.append(actual_peaks[line])
          item = actual_peaks[line]

  
  final_line_images = []
  start = 0
  for i in new_lines:
    final_line_images.append(img[start:i, :])
    start = i


  # for each line 
  final_word_images = []
  for line_image in final_line_images:
    
    col_hist = np.sum(line_image, axis=0).tolist()
    maximum = np.max(col_hist)
    diff = np.diff(col_hist)
    gradient = np.sign(diff)
    locations = np.where(gradient == 0)[0]
    locations = np.where(col_hist>0.99*maximum)[0]
    
    newloc = locations  

    # clustering: dbscan
    width_of_image = len(line_image[0])
    lasson = pd.DataFrame(newloc, columns=['pizza'])

    dbscan=DBSCAN(eps=5, min_samples=5)
    dbscan.fit(lasson)

    lasson['labels'] = dbscan.labels_
    lasson_drop = lasson[lasson['labels']!= -1]
    lasson_drop.reset_index(inplace = True)

    new_lasson = np.array(lasson_drop.groupby(lasson_drop.labels)[['pizza']].median()['pizza'], dtype='int')


    # cropping page to line

    
    start = new_lasson[0]
    for i in new_lasson[1:]:
      final_word_images.append(line_image[:, start:int(i)])
      start = int(i)

    pav = len(line_image[0])-1
    



    final_word_images.append(line_image[:, start:pav])  

  return final_word_images
  

  



    # here the output should be list of recognized words

    # PLOTTIGN ALL CROPPED IMAGES
    # n = len(final_word_images)
    # count = 0
    # for i in final_word_images:
    #     print(i)


characters = {'!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
characters = sorted(characters)

AUTOTUNE = tf.data.AUTOTUNE
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
print(char_to_num)

# hyper parameters
padding_token = 99
image_width = 128
image_height = 32
max_len=21

class CTCLayer(keras.layers.Layer):
  def __init__(self, name=None, **kwargs):
    super().__init__(name=name)
    self.loss_fn = keras.backend.ctc_batch_cost

  def call(self, y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = self.loss_fn(y_true, y_pred, input_length, label_length)
    self.add_loss(loss)

    # At test time, just return the computed predictions.
    return y_pred



def distortion_free_resize(image, img_size):
  w, h = img_size
  image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

  # Check tha amount of padding needed to be done.
  pad_height = h - tf.shape(image)[0]
  pad_width = w - tf.shape(image)[1]

  # Only necessary if you want to do same amount of padding on both sides.
  if pad_height % 2 != 0:
      height = pad_height // 2
      pad_height_top = height + 1
      pad_height_bottom = height
  else:
      pad_height_top = pad_height_bottom = pad_height // 2

  if pad_width % 2 != 0:
      width = pad_width // 2
      pad_width_left = width + 1
      pad_width_right = width
  else:
      pad_width_left = pad_width_right = pad_width // 2

  image = tf.pad(
      image,
      paddings=[
          [pad_height_top, pad_height_bottom],
          [pad_width_left, pad_width_right],
          [0, 0],
      ],
  )

  image = tf.transpose(image, perm=[1, 0, 2])
  image = tf.image.flip_left_right(image)
  return image


def preprocess_image(image_path, img_size=(image_width, image_height)):
  # image = tf.io.read_file(image_path)
  # image = tf.image.decode_png(image, 1)
  
  #image = tf.keras.preprocessing.image.array_to_img(img)
  cc = image_path.shape
  image = image_path.reshape((cc[0], cc[1], 1))

  image = tf.convert_to_tensor(image, dtype=tf.float32)
  #image = tf.image.decode_png(image, 1)

  #image = image_path

  image = distortion_free_resize(image, img_size)
  image = tf.cast(image, tf.float32) / 255.0
  return image

def decode_batch_predictions(pred):
  input_len = np.ones(pred.shape[0]) * pred.shape[1]
  # Use greedy search. For complex tasks, you can use beam search.
  results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
      :, :max_len
  ]
  # Iterate over the results and get back the text.
  output_text = []
  for res in results:
      res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
      res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
      output_text.append(res)
  return output_text

def gaurav(wordimage, prediction_model, model):
  img = preprocess_image(wordimage)
  img = np.array(img)
  img = img[None,:,:,:]

  preds = prediction_model.predict(img)
  pred_texts = decode_batch_predictions(preds)
  print(pred_texts[0])
  return pred_texts[0] + ' '



'''old recognition module'''
# def wordrecog(word_images):
#   model = tf.keras.models.load_model("home/handwritten_text.h5", custom_objects={'CTCLayer': CTCLayer})
#   prediction_model = keras.models.Model(model.get_layer(name="image").input, model.get_layer(name="dense2").output)
#   coffee = []
#   for word_image in word_images:
#     coffee.append(gaurav(word_image, prediction_model, model))
  
#   tea = ''.join(coffee) # tea is string version of all words in a page
#   return tea


# new one
def wordrecog(word_images):
  # loading the model with the trained weights
  weights = 'home/two-weights-epoch32-acc0.943-val_acc0.821--loss0.148-val_loss1.261.h5'
  model = make_model(weights)

  # now using the model to make predictions of mulitple files
  coffee = []
  for word_image in word_images:
    coffee.append(make_predictions(word_image, model))
  
  tea = ''.join(coffee) # tea is string version of all words in a page
  return tea






def speller(text):
  # magic
  corrected_text = spell(text)
  return corrected_text

def doc2vec(docs):
  list_vectors_doc = []
  return list_vectors_doc

def plagia(doc1, doc2):
  # cosine similarity
  return 0.5




def process_tfidf_similarity(documents):
  vectorizer = TfidfVectorizer()
  embeddings = vectorizer.fit_transform(documents)
  cosine_similarities = cosine_similarity(embeddings[:], embeddings[:]).flatten()
  result = []
  for sim in cosine_similarities:
    result.append(round(sim,2))
  return np.array(result)


def tfidf_answerkey(documents, answerkey):
  vectorizer = TfidfVectorizer()
  documents.append(answerkey)
  embeddings = vectorizer.fit_transform(documents)
  cosine_similarities = cosine_similarity(embeddings[-1], embeddings[:-1]).flatten()
  result = []
  for sim in cosine_similarities:
    result.append(round(sim,2))
  return np.array(result)

def make_model(weights):
  input_data = layers.Input(name='the_input', shape=(128, 64, 1), dtype='float32')
  iam_layers = layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
  iam_layers = layers.BatchNormalization()(iam_layers)
  iam_layers = layers.Activation('relu')(iam_layers)
  iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(iam_layers)

  iam_layers = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(iam_layers)
  iam_layers = layers.BatchNormalization()(iam_layers)
  iam_layers = layers.Activation('relu')(iam_layers)
  iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(iam_layers)

  iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(iam_layers)
  iam_layers = layers.BatchNormalization()(iam_layers)
  iam_layers = layers.Activation('relu')(iam_layers)

  iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(iam_layers)
  iam_layers = layers.BatchNormalization()(iam_layers)
  iam_layers = layers.Activation('relu')(iam_layers)
  iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(iam_layers)

  iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(iam_layers)
  iam_layers = layers.BatchNormalization()(iam_layers)
  iam_layers = layers.Activation('relu')(iam_layers)

  iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv6')(iam_layers)
  iam_layers = layers.BatchNormalization()(iam_layers)
  iam_layers = layers.Activation('relu')(iam_layers)
  iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(iam_layers)

  iam_layers = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(iam_layers)
  iam_layers = layers.BatchNormalization()(iam_layers)
  iam_layers = layers.Activation('relu')(iam_layers)

  iam_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(iam_layers)
  iam_layers = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(iam_layers)
  iam_layers = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(iam_layers)

  iam_layers = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(iam_layers)
  iam_layers = layers.BatchNormalization()(iam_layers)

  iam_layers = layers.Dense(80, kernel_initializer='he_normal', name='dense2')(iam_layers)
  iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)

  # Creating the Model
  
  model = Model(inputs=input_data, outputs=iam_outputs)
  model.load_weights(filepath=weights)
  # print(model.summary())

  return model

def add_padding(img, old_w, old_h, new_w, new_h):

    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    img_pad = np.ones([new_h, new_w, 1]) * 255
    img_pad[h1:h2, w1:w2, 0] = img
    return img_pad

def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w < target_w and h < target_h:
        new_img = cv2.resize(img, (w, h),interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, w, h, target_w, target_h)
    elif w >= target_w and h < target_h:
        new_w = target_w
        new_h = int(h * new_w / w)
        new_img = cv2.resize(img, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w < target_w and h >= target_h:
        new_h = target_h
        new_w = int(w * new_h / h)
        new_img = cv2.resize(img, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        """w>=target_w and h>=target_h """
        ratio = max(w / target_w, h / target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv2.resize(img, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img


def preprocess(path, img_w, img_h):
    """ Pre-processing image for predicting """

    img = path
    img = fix_size(img, img_w, img_h)

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = img.reshape((img_h, img_w))
    img = img.astype(np.float32)
    img /= 255
    return img

def numbered_array_to_text(numbered_array):

    letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    numbered_array = numbered_array[numbered_array != -1]
    return "".join(letters[i] for i in numbered_array)

def make_predictions(pa, model):

    test_images_processed = []
    sh = pa.shape
    pa = pa.reshape((sh[0], sh[1], 1))
    temp_processed_image = preprocess(path=pa, img_w=128, img_h=64)
    test_images_processed.append(temp_processed_image.T)
    test_images_processed = np.array(test_images_processed)
    test_predictions_encoded = model.predict(x=test_images_processed)
    test_predictions_decoded = tf_keras_backend.get_value(tf_keras_backend.ctc_decode(test_predictions_encoded, input_length=np.ones(test_predictions_encoded.shape[0])*test_predictions_encoded.shape[1], greedy=True)[0][0])
    res = numbered_array_to_text(test_predictions_decoded).replace('7', '')
    print(res)
    return res
    #self.store.append(self.numbered_array_to_text(test_predictions_decoded))