from unittest import result
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras import backend as tf_keras_backend


class Predictions:

    def __init__(self, link):

        p = Path(link)
        self.make_model()
        self.store = []
        self.result = []
        for pa in p.iterdir():
            self.make_predictions(str(pa))

        self.result.append(self.store)

    def make_model(self):

        input_data = layers.Input(
            name='the_input', shape=(128, 64, 1), dtype='float32')
        iam_layers = layers.Conv2D(
            64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.MaxPooling2D(
            pool_size=(2, 2), name='max1')(iam_layers)

        iam_layers = layers.Conv2D(
            128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.MaxPooling2D(
            pool_size=(2, 2), name='max2')(iam_layers)

        iam_layers = layers.Conv2D(
            256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)

        iam_layers = layers.Conv2D(
            256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.MaxPooling2D(
            pool_size=(1, 2), name='max3')(iam_layers)

        iam_layers = layers.Conv2D(
            512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)

        iam_layers = layers.Conv2D(
            512, (3, 3), padding='same', name='conv6')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)
        iam_layers = layers.MaxPooling2D(
            pool_size=(1, 2), name='max4')(iam_layers)

        iam_layers = layers.Conv2D(
            512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)
        iam_layers = layers.Activation('relu')(iam_layers)

        iam_layers = layers.Reshape(target_shape=(
            (32, 2048)), name='reshape')(iam_layers)
        iam_layers = layers.Dense(
            64, activation='relu', kernel_initializer='he_normal', name='dense1')(iam_layers)
        iam_layers = layers.Bidirectional(layers.LSTM(
            units=256, return_sequences=True))(iam_layers)

        iam_layers = layers.Bidirectional(layers.LSTM(
            units=256, return_sequences=True))(iam_layers)
        iam_layers = layers.BatchNormalization()(iam_layers)

        iam_layers = layers.Dense(
            80, kernel_initializer='he_normal', name='dense2')(iam_layers)
        iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)

        # Creating the Model
        self.iam_model_pred = None
        self.iam_model_pred = Model(inputs=input_data, outputs=iam_outputs)

    def add_padding(self, img, old_w, old_h, new_w, new_h):

        h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
        w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
        img_pad = np.ones([new_h, new_w, 3]) * 255
        img_pad[h1:h2, w1:w2, :] = img
        return img_pad

    def fix_size(self, img, target_w, target_h):
        h, w = img.shape[:2]
        if w < target_w and h < target_h:
            img = self.add_padding(img, w, h, target_w, target_h)
        elif w >= target_w and h < target_h:
            new_w = target_w
            new_h = int(h * new_w / w)
            new_img = cv2.resize(img, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_w, target_h)
        elif w < target_w and h >= target_h:
            new_h = target_h
            new_w = int(w * new_h / h)
            new_img = cv2.resize(img, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_w, target_h)
        else:
            """w>=target_w and h>=target_h """
            ratio = max(w / target_w, h / target_h)
            new_w = max(min(target_w, int(w / ratio)), 1)
            new_h = max(min(target_h, int(h / ratio)), 1)
            new_img = cv2.resize(img, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            img = self.add_padding(new_img, new_w, new_h, target_w, target_h)
        return img

    def preprocess(self, path, img_w, img_h):
        """ Pre-processing image for predicting """
        img = cv2.imread(str(path))
        img = self.fix_size(img, img_w, img_h)

        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float32)
        img /= 255
        return img

    def numbered_array_to_text(self, numbered_array):

        letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                   'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        numbered_array = numbered_array[numbered_array != -1]
        return "".join(letters[i] for i in numbered_array)

    def make_predictions(self, pa):

        self.iam_model_pred.load_weights(
            filepath=r'C:\Users\gaura\OneDrive\Desktop\Handwriting_Classification\two-weights-epoch32-acc0.943-val_acc0.821--loss0.148-val_loss1.261.h5')
        test_images_processed = []
        temp_processed_image = self.preprocess(path=pa, img_w=128, img_h=64)
        test_images_processed.append(temp_processed_image.T)
        test_images_processed = np.array(test_images_processed)
        test_predictions_encoded = self.iam_model_pred.predict(
            x=test_images_processed)
        test_predictions_decoded = tf_keras_backend.get_value(tf_keras_backend.ctc_decode(test_predictions_encoded, input_length=np.ones(
            test_predictions_encoded.shape[0])*test_predictions_encoded.shape[1], greedy=True)[0][0])
        print(self.numbered_array_to_text(
            test_predictions_decoded))
        self.store.append(self.numbered_array_to_text(
            test_predictions_decoded))
