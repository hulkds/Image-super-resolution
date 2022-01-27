from data_extraction import data_extraction
from model import SRmodel
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
import os

def inference(model_path, test_data_path):
    """inference using our best model: ks=5, kr=3, filters=64, lr=0.02

    Args:
        model_path (str): path to the trained model.
        test_data_path (str): path to the test data.
    """    
    model = SRmodel(ks=5, kr=3, filters=64, lr=0.02)
    model.load_weights(model_path)

    test_HR, test_LR = data_extraction(path=test_data_path)

    test_SR = model.predict(test_LR)

    for name in ['LR', 'HR', 'SR']:
        if not os.path.exists('savepath/' + name):
            os.makedirs('savepath/' + name)

    for i in range(test_LR.shape[0]):
        path_LR = 'savepath/LR/' + str(i) + '.png'
        path_HR = 'savepath/HR/' + str(i) + '.png'
        path_SR = 'savepath/SR/' + str(i) + '.png'
        
        Image.fromarray((255.*np.clip(test_LR[i,:,:,:], 0., 1.)).astype('uint8')).save(path_LR)
        Image.fromarray((255.*np.clip(test_HR[i,:,:,:], 0., 1.)).astype('uint8')).save(path_HR)
        Image.fromarray((255.*np.clip(test_SR[i,:,:,:], 0., 1.)).astype('uint8')).save(path_SR)

def evaluate(model_path, test_data_path, train_data_path, val_data_path, batch_size):
    model = SRmodel(ks=5, kr=3, filters=64, lr=0.02)
    model.load_weights(model_path)

    test_HR, test_LR = data_extraction(path=test_data_path) #test data
    train_HR, train_LR = data_extraction(path=train_data_path) #training data
    val_HR, val_LR = data_extraction(path=val_data_path) #validation data

    train_SR = model.predict(train_LR, batch_size=batch_size)
    val_SR = model.predict(val_LR, batch_size=batch_size)
    test_SR = model.predict(test_LR)

    train_psnr = tf.math.reduce_mean(tf.image.psnr(train_SR, train_HR, max_val=1.))
    val_psnr = tf.math.reduce_mean(tf.image.psnr(val_SR, val_HR, max_val=1.))
    test_psnr = tf.math.reduce_mean(tf.image.psnr(test_SR, test_HR, max_val=1.))
    print('train_psnr: ', K.get_value(train_psnr))
    print('val_psnr: ', K.get_value(val_psnr))
    print('test_psnr: ', K.get_value(test_psnr))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test super resolution trained model.')
    parser.add_argument('--mode', type=str, default='evaluate',
                        help='evaluate mode or inference mode.')
    parser.add_argument('--trainDataPath', type=str, default='train_data/',
                        help='path to the training data.')
    parser.add_argument('--valDataPath', type=str, default='val_data/',
                        help='path to the validation data.')
    parser.add_argument('--testDataPath', type=str, default='test_data/',
                        help='path to the test data.')
    parser.add_argument('--modelWeightsPath', type=str, default='trained/SRModel.hdf5',
                        help='path to the trained weights.')
    parser.add_argument('--batchSize', type=int, default=16,
                        help='batch size.')

    args = parser.parse_args()

    if(args.mode == 'evaluate'):
        evaluate(model_path=args.modelWeightsPath, test_data_path=args.testDataPath, train_data_path=args.trainDataPath, val_data_path=args.valDataPath, batch_size=args.batchSize)
    elif(args.mode == 'inference'):
        inference(model_path=args.modelWeightsPath, test_data_path=args.testDataPath)
    else:
        raise AttributeError(f'Undefined mode: {args.mode}')

