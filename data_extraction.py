import numpy as np
import tensorflow as tf
from PIL import Image
from os import listdir
from os.path import join, isfile
import cv2

def data_extraction(path='./train_data/', size=256):
    """extract patches from original image to the size of 256x256 for HR images and then, downsample the HR image to produce LR image with size of 128x128.

    Args:
        path (str, optional): path to the training dataset. 
        size (int, optional): size of the patches.

    Returns:
        HR_image (np.array): HR images of shape (.,256,256,3), used to train.
        HR_image (np.array): LR images of shape (.,128,128,3), used to train.
    """    
    files = [path + img_f for img_f in listdir(path) if isfile(join(path, img_f))]
    data = []
    exception_count = 1
    for f in files:
        try:
            img = np.asarray(Image.open(f))
            nx, ny = img.shape[0]//size, img.shape[1]//size
            for i in range(nx):
                for j in range(ny):
                    extracted_img = np.reshape(img[size*i:size*(i+1), size*j:size*(j+1), :], (1,size,size,3))
                    data.append(extracted_img)
        except:
            print("exception" ,exception_count, ":", f)
            exception_count +=1
            continue
    
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)
    
    #HR ground truth train/val/test images
    HR_img = data.astype('float32')/255.
    #HR_img_test = data[3540:, :, :, :].astype('float32')/255.
    
    #LR train/val/test images
    #use maxpool2d of tensorflow to generate LR images from HR images
    data = tf.convert_to_tensor(data)
    data = tf.nn.max_pool(data, ksize=(2,2), strides=2, padding='VALID')
    data = np.asarray(data)
    
    LR_img = data.astype('float32')/255.
    
    return HR_img, LR_img

if __name__ == "__main__":
    HR_img, LR_img = data_extraction()
    print(HR_img.shape)
    print(LR_img.shape)
        
        
        
        
        
        
        
        
        