import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from data_extraction import data_extraction
from utils import save_history

import os

def train(base_name, model, train_data_path, val_data_path, batchs_size, epochs, early_stopping):
    """train super resolution model

    Args:
        base_name (str): base name to save the trained model.
        model (Keras model): Super resolution keras model.
        train_data_path (str): path to the training data.
        val_data_path (str): path to the validation data.
        batchs_size (int): batch size.
        epochs (int): number of epochs.
        early_stopping (boolean): either using early stopping or not.
    """    
    if not os.path.exists('trained/'):
        os.makedirs('trained/')

    filepath = 'trained/' + base_name + '.hdf5'

    # save the best model if there is
    saving = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                save_weights_only=True)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3) # learning rate scheduler

    train_HR, train_LR = data_extraction(path=train_data_path) #training data
    val_HR, val_LR = data_extraction(path=val_data_path) #validation data

    # using early stopping or not
    if early_stopping:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        callbacks = [lr_scheduler, early_stopping_cb, saving]
    else:
        callbacks = [lr_scheduler, saving]

    # train model
    history = model.fit(train_LR, train_HR,
                        epochs=epochs,
                        batch_size=batchs_size,
                        validation_data=(val_LR, val_HR),
                        verbose=1,
                        callbacks=callbacks
    )   
    
    # save training history
    save_history(base_name, history)

if __name__ == '__main__':
    from model import SRmodel
    import argparse

    parser = argparse.ArgumentParser(description='Training super resolution model option.')
    parser.add_argument('--numResBlock', type=int, default=5,
                        help='number of residual block in the model.')
    parser.add_argument('--numConvBlock', type=int, default=3,
                        help='number of convolution base block in the dense block.')
    parser.add_argument('--filters', type=int, default=64,
                        help='number of base filters.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='learning rate.')
    parser.add_argument('--trainDataPath', type=str, default='train_data/',
                        help='path to the training data.')
    parser.add_argument('--valDataPath', type=str, default='val_data/',
                        help='path to the validation data.')
    parser.add_argument('--batchSize', type=int, default=16,
                        help='Batch size.') 
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs.')                   
    parser.add_argument('--earlyStopping', type=bool, default=True,
                        help='use early stopping or not.') 
    
    args = parser.parse_args()

    model = SRmodel(ks=args.numResBlock, kr=args.numConvBlock, filters=args.filters, lr=args.lr)
    train(base_name='SRModel', model=model, train_data_path=args.trainDataPath, val_data_path=args.valDataPath, batchs_size=args.batchSize, 
            epochs=args.epochs, early_stopping=args.earlyStopping)