import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def conv2dbase(x, filters=96, kernel_size=3, strides=1, padding='same', alpha=0.2):
    """implementing of the base convolution: conv2d -> batchnorm -> LeakyRELU.

    Args:
        x (tensor): input tensor.
        filters (int, optional): number of filters.
        kernel_size (int, optional): kernel size.
        strides (int, optional): stride.
        padding (str, optional): padding 'same' or 'valid'.
        alpha (float, optional): alpha parameter used in LeakyRELU activation. 

    Returns:
        [tensor]: ouput tensor.
    """    
    y = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    y = BatchNormalization()(y)
    y = LeakyReLU(alpha=alpha)(y)

    return y

def denseblock(x, filters=96, k=3):
    """implemention of denseblock presented in the paper.

    Args:
        x (tensor): input tensor.
        filters (int, optional): number of filters.
        k (int, optional): number of convolution base block in dense block.

    Returns:
        [tensor]: output tensor.
    """    
    for i in range(k):
        x_temp = conv2dbase(x, filters=filters, kernel_size=3)
        x = Concatenate(axis=-1)([x, x_temp])
    
    #do this for higher parameter and computational effciciency    
    if(k > 3):
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = conv2dbase(x, filters=filters*4, kernel_size=1)
        
    x = conv2dbase(x, filters=filters, kernel_size=3)
        
    return x

def rdb(x, filters=96, kr=3):
    """implemention of the residual dense block presented in the paper.

    Args:
        x (tensor): input tensor.
        filters (int, optional): number of filters.
        kr (int, optional): number of convolution base block in the dense block.

    Returns:
        [tensor]: output tensor.
    """    
    #y = conv2dbase(x, filters=64, kernel_size=3)
    y = denseblock(x, filters=filters, k=kr)
    y = Concatenate(axis=-1)([y, x])
    y = conv2dbase(y, filters=filters, kernel_size=3)
    y = Add()([y, x])
    
    return y

def SRmodel(ks=5, kr=3, filters=64, lr=0.02):
    """implemention of super resolution model.

    Args:
        ks (int, optional): number of residual block in the model.
        kr (int, optional): number of convolution base block in the  dense block.
        filters (int, optional): number of the filters.
        lr (float, optional): learning rate.

    Returns:
        [Model]: keras model.
    """    
    resb = []
    inputs = Input(shape=(128, 128, 3))
    x = conv2dbase(inputs, filters=filters, kernel_size=3)
    
    resb.append(x)
    for i in range(ks):
        resb.append(rdb(resb[i], kr=kr, filters=filters))
    resb = resb[1:]
    concat = Concatenate(axis=-1)(resb)
    
    y = conv2dbase(concat, filters=filters, kernel_size=1)
    y = conv2dbase(concat, filters=filters, kernel_size=3)
    
    y = Add()([y, x])
    y = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(y)
    #y = conv2dbase(y, filters=32, kernel_size=1)
    
    outputs = conv2dbase(y, filters=3, kernel_size=3, alpha=0.0)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss='mse')
    
    return model    
    
if __name__ == '__main__':
    model = SRmodel()
    model.summary()
    
    
    
    
    
    
    
    
    
    
