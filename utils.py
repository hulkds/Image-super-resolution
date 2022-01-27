from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ExponentialLearningRate(keras.callbacks.Callback):
    """exponential learning rate scheduler used to find the optimal zone of learning rate.
    """    
    def __init__(self, factor):
        self.factor = factor
        self.rates = [] # list of learning rate.
        self.losses = [] # list of losses corresponding to every lr values.
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor) # update learning rate.

def find_learning_rate(model, X, y, epochs=1, batch_size=2, min_rate=10**-5, max_rate=10):
    """find the optimal zone of the learning rate.

    Args:
        model (Keras Model): super resolution keras model.
        X (tensor): input tensor images.
        y (tensor): output tensor images.
        epochs (int, optional): number of epochs.
        batch_size (int, optional): batch size.
        min_rate (float, optional): minimum value of learning rate.
        max_rate (int, optional): minimum value of learning rate.

    Returns:
        rates (list): list of learning rates.
        losses (list): list of losses.
    """    
    init_weights = model.get_weights()
    iterations = len(X) // batch_size * epochs 
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    """plot the learning rate in the function of losses.

    Args:
        rates ([list]): list of learning rate.
        losses ([list]): list of losses.
    """    
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig("optimal_lr.png")

def save_history(base_name, history):
    """save training history

    Args:
        base_name (str): base file name.
        history (list): list of history.
    """    
    base_name = 'SRModel' 
    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    base_file = 'trained/' + base_name + '.json' 
    with open(base_file, mode='w') as f:
        hist_df.to_json(f)

if __name__ == '__main__':
    from data_extraction import data_extraction
    from model import SRmodel

    train_HR, train_LR = data_extraction('train_data/')
    val_HR, val_LR = data_extraction('val_data/')
    model = SRmodel()
    rates, losses = find_learning_rate(model, train_LR, train_HR)