import socketio
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.compat.v1.keras import backend as K
import numpy as np
import os
import pandas as pd
import codecs
import pickle



import glob
from imutils import paths

import msgpack
import msgpack_numpy as m

def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)

def json_serialize(weights):
    serialized_weights = lambda a: [i.tolist() for i in a]
    return serialized_weights(weights)

def json_deserialize(weights):
    deserialized_weights = lambda a: [np.array(i) for i in a]
    return deserialized_weights(weights)

def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """
    
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    
    return positive_frequencies, negative_frequencies

def load(image_path, verbose=-1):
    '''expects images for each class in seperate dir, 
    e.g all digits in 0 class in the directory named 0 
    For the first time training on a data enter image data path.
    Otherwise enter the csv file path'''
    #data = list()
    #labels = list()
    csv_files = glob.glob(os.path.join(image_path, '*.csv'))
    # loop over the input images
    if(len(csv_files)==1):
        print(csv_files)
        return pd.read_csv(csv_files[0])
    image_paths = list(paths.list_images(image_path))
    for (i, imgpath) in enumerate(image_paths):
        # load the image and extract the class labels
        #im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        #image = np.array(im_gray)
        label = np.float32(imgpath.split(os.path.sep)[-2].split("/")[-1])
        if(i==0):
            dataframe = pd.DataFrame({"Image": imgpath, "Covid": label, "Normal": abs(label-1)}, index=[i])
        else:
            dataframe = dataframe.append({"Image": imgpath, "Covid": label, "Normal": abs(label-1)}, ignore_index=True)
        # scale the image to [0, 1] and add to list
        #data.append(image/255)
        #labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))
    # return a tuple of the data and labels
    #return data, labels, dataframe
    dataframe.to_csv(f'{image_path}/dataframe.csv')
    print(f"The file dataset.csv has been created, use {image_path}/dataframe.csv the next time to save your time")
    return dataframe

def get_train_generator(df, image_dir, x_col, y_cols, shuffle=True, batch_size=38, seed=1, target_w = 32, target_h = 32):
    """
    Return generator for training set, normalizing using batch
    statistics.

    Args:
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        train_generator (DataFrameIterator): iterator over training set
    """        
    print("getting train generator...") 
    # normalize images
    image_generator = ImageDataGenerator(
        rescale = 1./255,
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    # flow from directory with specified batch size
    # and target image size
    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            color_mode='grayscale',
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            target_size=(target_w,target_h))
    
    return generator

def get_test_and_valid_generator(valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=32, seed=1, target_w = 32, target_h = 32):
    """
    Return generator for validation set and test test set using 
    normalization statistics from training set.

    Args:
      valid_df (dataframe): dataframe specifying validation data.
      test_df (dataframe): dataframe specifying test data.
      train_df (dataframe): dataframe specifying training data.
      image_dir (str): directory where image files are held.
      x_col (str): name of column in df that holds filenames.
      y_cols (list): list of strings that hold y labels for images.
      sample_size (int): size of sample to use for normalization statistics.
      batch_size (int): images per batch to be fed into model during training.
      seed (int): random seed.
      target_w (int): final width of input images.
      target_h (int): final height of input images.
    
    Returns:
        test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
    """
    print("getting train and valid generators...")
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        directory=image_dir, 
        x_col=x_col, 
        y_col=y_cols,
        class_mode="raw", 
        color_mode="grayscale",
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        rescale = 1./255,
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            directory=image_dir,
            color_mode='grayscale',
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            color_mode='grayscale',
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator






