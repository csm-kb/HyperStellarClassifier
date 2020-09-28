import os
import datetime
from time import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class CNN:
    def __init__(self, layers, optimizer, loss, metrics, verbose=False):
        # init simple vars
        self.verbose = verbose
        self.metrics = metrics
        self.optimizer = optimizer
        self.loss = loss

        # create the model!
        # The lines of code below define the convolutional base using a common pattern: a stack of various tf.keras.Layer objects:
        #   - As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size.
        #   - The dim 'color_channels' refers to (R,G,B), typically 3.
        self.model = tf.keras.models.Sequential()

        for layer in layers:
            name, args = layer
            assert(name)
            assert(isinstance(args, dict))
            assert(name in ['Conv2D', 'MaxPooling2D', 'BatchNorm', 'Dense', 'Dropout', 'Activation', 'Flatten', 'GlobalMaxPooling2D'])
            if name == 'Conv2D':
                self.model.add(tf.keras.layers.Conv2D(**args))
            elif name == 'MaxPooling2D':
                self.model.add(tf.keras.layers.MaxPooling2D(**args))
            elif name == 'BatchNorm':
                self.model.add(tf.keras.layers.BatchNormalization(**args))
            elif name == 'Dense':
                self.model.add(tf.keras.layers.Dense(**args))
            elif name == 'Dropout':
                self.model.add(tf.keras.layers.Dropout(**args))
            elif name == 'Activation':
                self.model.add(tf.keras.layers.Activation(**args))
            elif name == 'Flatten':
                self.model.add(tf.keras.layers.Flatten(**args))
            elif name == 'GlobalMaxPooling2D':
                self.model.add(tf.keras.layers.GlobalMaxPooling2D(**args))

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.model.summary()

    def fit(self, images, labels, epochs, validation_data):
        # train the model
        history = self.model.fit(images, labels, epochs=epochs, validation_data=validation_data)

        # evaluate/plot the model
        plt.plot(history.history['root_mean_squared_error'], label='root_mean_squared_error')
        plt.plot(history.history['val_root_mean_squared_error'], label = 'val_root_mean_squared_error')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        if not os.path.exists('./out/{}'.format(self.__name__)):
            os.makedirs('./out/{}'.format(self.__name__))
        now = datetime.datetime.now()
        plt.savefig('./out/{}/plot_epoch_rmse-{}{}{}_{}{}{}.png'.format(self.__name__,now.month,now.day,now.year,now.hour,now.minute,now.second))

        test_images, test_labels = validation_data
        test_loss, test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)
        print('loss\t: {0}\nrmse\t: {1}'.format(test_loss, test_acc))
        # save eval results to csv
        out_df = pd.DataFrame(np.array([test_loss, test_acc]), columns=['loss','rmse'])
        out_df.to_csv('./out/{}/eval_loss_rmse-{}{}{}_{}{}{}.csv'.format(self.__name__,now.month,now.day,now.year,now.hour,now.minute,now.second), index=False)

    def predict(self, images):
        return self.model.predict(images)
