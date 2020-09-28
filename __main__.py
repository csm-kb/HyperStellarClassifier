import os
import shutil
import random
import time
import datetime
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

import models
from models import cnn

def root_mean_squared_error(y_true, y_pred):
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

ORIG_SHAPE = (424,424)
CROP_SIZE = (256,256)
IMG_SHAPE = (64,64)

data_dir = r'./data'
training_img_cache  = os.path.join(data_dir, r'images_training_cache')
training_imgs       = os.path.join(data_dir, r'images_training_rev1')
test_imgs           = os.path.join(data_dir, r'images_test_rev1')

def get_image_cache(path):
    x = plt.imread(path)
    x = x/255.
    return x

def get_image(path, x1,y1, shape, crop_size):
    x = plt.imread(path)
    x = x[x1:x1+crop_size[0], y1:y1+crop_size[1]]
    x = resize(x, shape)
    x = x/255.
    return x
    
def get_all_images(dataframe, img_dir, img_type, shape=IMG_SHAPE, crop_size=CROP_SIZE):
    x1 = (ORIG_SHAPE[0]-CROP_SIZE[0])//2
    y1 = (ORIG_SHAPE[1]-CROP_SIZE[1])//2
   
    sel = dataframe.values
    ids = sel[:,0].astype(int).astype(str)
    y_batch = sel[:,1:]
    x_batch = []

    print('Loading "'+img_type+'" images:')
    for i in tqdm(ids):
        x = get_image(os.path.join(img_dir, str(i)+'.jpg'), x1, y1, shape=shape, crop_size=crop_size)
        x_batch.append(x)
    x_batch = np.array(x_batch)
    return x_batch, y_batch

def test_image_generator(ids, test_dir, shape=IMG_SHAPE):
    x1 = (ORIG_SHAPE[0]-CROP_SIZE[0])//2
    y1 = (ORIG_SHAPE[1]-CROP_SIZE[1])//2
    x_batch = []
    for i in ids:
        x = get_image(os.path.join(test_dir, str(i)), x1, y1, shape=IMG_SHAPE, crop_size=CROP_SIZE)
        x_batch.append(x)
    x_batch = np.array(x_batch)
    return x_batch

if __name__ == "__main__":
    assert(os.path.exists(data_dir))
    startTime = time.time()

    # this is set up so you can specify multiple different models or variations of models, for direct comparison, at the same time!
    #   - note: a single 'models.cnn.CNN' with this config + 30 epochs takes ~1.5+ hours to train (~2 min/epoch), so beware...
    kargs_dict_list = [
        {'class': models.cnn.CNN,
        '__init__': {
            'layers':[
            ('Conv2D', { 'filters':512,'kernel_size':(3, 3),'input_shape':(IMG_SHAPE[0], IMG_SHAPE[1], 3) }),
            ('Conv2D', { 'filters':256,'kernel_size':(3, 3) }),
            ('Activation', { 'activation':'relu' }),
            ('MaxPooling2D', { 'pool_size':(2, 2) }),

            ('Conv2D', { 'filters':256,'kernel_size':(3, 3) }),
            ('Conv2D', { 'filters':128,'kernel_size':(3, 3) }),
            ('Activation', { 'activation':'relu' }),
            ('MaxPooling2D', { 'pool_size':(2, 2) }),

            ('Conv2D', { 'filters':128,'kernel_size':(3, 3) }),
            ('Conv2D', { 'filters':128,'kernel_size':(3, 3) }),
            ('Activation', { 'activation':'relu' }),
            ('GlobalMaxPooling2D', {}),

            ('Dropout', { 'rate':0.25 }),
            ('Dense', { 'units':128 }),
            ('Activation', { 'activation':'relu' }),
            ('Dropout', { 'rate':0.25 }),
            ('Dense', { 'units':128 }),
            ('Activation', { 'activation':'relu' }),
            ('Dropout', { 'rate':0.25 }),
            ('Dense', { 'units':128 }),
            ('Activation', { 'activation':'relu' }),
            ('Dropout', { 'rate':0.25 }),
            ('Dense', { 'units':37 }),
            ('Activation', { 'activation':'sigmoid' })
            ],
            'optimizer':'adamax', 'loss':'binary_crossentropy', 'metrics':[root_mean_squared_error]
            },
        'epochs': 30
        }
    ]

    df = pd.read_csv(os.path.join(data_dir, r'training_solutions_rev1.csv'), encoding='utf-8')

    df_train, df_test = train_test_split(df, test_size=0.2)
    print('df_train size:\t{0}\ndf_test size:\t{1}'.format(str(df_train.shape), str(df_test.shape)))
    
    X_train, y_train = get_all_images(df_train, training_imgs, 'train')
    X_test, y_test = get_all_images(df_test, training_imgs, 'test')

    # iterate through each model we defined in our dict earlier
    for kargs_dict in kargs_dict_list:
        now = datetime.datetime.now()
        model = kargs_dict['class'](**kargs_dict['__init__'])
        model_out_dir = './out/{}'.format(model.__name__)
        assert(model)
        batch_size = 128
        # train the model
        model.fit(X_train, y_train, epochs=kargs_dict['epochs'], validation_data=(X_test, y_test))
        # once done, prepare the validation dataset
        val_files = os.listdir(test_imgs)
        val_preds = []
        N_val = len(val_files)
        for i in tqdm(np.arange(0, N_val, batch_size)):
            if batch_size + i > N_val:
                upper = N_val
            else:
                upper = batch_size + i
            X = test_image_generator(val_files[i:upper], test_imgs)
            # predict classes for this image
            y_pred = model.predict(X)
            val_preds.append(y_pred)
        # prepare data for write
        val_preds = np.array(val_preds)
        Y_pred = np.vstack(val_preds)
        ids = np.array([v.split('.')[0] for v in val_files]).reshape(len(val_files), 1)
        out_df = pd.DataFrame(np.hstack((ids, Y_pred)), columns=df.columns)
        out_df = out_df.sort_values(by=['GalaxyID'])
        # write test results to file
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)
        out_title = 'sample_out-{}-{}{}{}_{}{}{}.csv'.format(model.__name__,now.month,now.day,now.year,now.hour,now.minute,now.second)
        out_df.to_csv(model_out_dir + out_title, index=False)

    print('Finished! Time elapsed: {} seconds'.format(str(time.time() - startTime)))
