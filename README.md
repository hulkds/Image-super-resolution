# Image-super-reso

Super-resolution is the task of constructing a higher-resolution image from a lower-resolution image. While this task has traditionally been approched with non-linear methodes such as bilinear and bicubic upsampling, neural networks offer an opportunity for significant improvements. 

Inspired from the Residual Dense Network for Image Restoration, the paper publiced at 23 Jan 2020 of Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu, Fellow, IEEE, i have implemented the Residual Dense Network for image super-resolution (RDN for image SR).
Link to the paper: [https://arxiv.org/pdf/1812.10477.pdf]

I draw the dataset from DIV2K, a dataset of 1000 high-resolution images with diverse subjects. Due to the constraints on GPU rescources, i used a subsample of 100 images, extracted to the size 256x256, so i got 2870 images for train and 780 images for validation. To generate lower-resolution images from higher-resolution images, i used max pooling method.

For the result i got the PSNR for the validation set is 31db after 54 epochs (using early stopping).
I have used my model to evaluate some classic images(lena, pepper...) but not the images from DIV2K, and i got 28db for PNSR. This score indicate that my best model can beat the bicubic upsampling for this images super-resolution task which just 23db. The result images is found in *savepath* folder.

# Requirements

- python version = 3.8.10
- tensorflow-gpu = 2.7.0
- opencv-python = 4.5.5.62

# Quick start

1. Install the required packages: 
    <pre><code>pip install pipenv.</code></pre>
    <pre><code>pipenv install.</code></pre>

2. Download the dataset: [https://drive.google.com/drive/folders/1WKOQC0JTtPSVm0INUc_PHqaQzz4_1r7S?usp=sharing]

3. Quick training with default options, trained model is saved in the folder `trained/`:
    <pre><code>pipenv run python train.py.</code></pre>

4. Quick evaluation with default options:
    <pre><code>pipenv run python test.py --mode evaluate.</code></pre>

5. Quick inference with default options, output images are saved in the folder `savepath/`
    <pre><code>pipenv run python test.py --mode inference.</code></pre>

# More options

1. Training:
<pre><code>usage: train.py [-h] [--numResBlock NUMRESBLOCK] [--numConvBlock NUMCONVBLOCK] [--filters FILTERS] [--lr LR] [--trainDataPath TRAINDATAPATH] [--valDataPath VALDATAPATH] [--batchSize BATCHSIZE]
                [--epochs EPOCHS] [--earlyStopping EARLYSTOPPING]

Training super resolution model option.

optional arguments:
  -h, --help            show this help message and exit
  --numResBlock NUMRESBLOCK
                        number of residual block in the model.
  --numConvBlock NUMCONVBLOCK
                        number of convolution base block in the dense block.
  --filters FILTERS     number of base filters.
  --lr LR               learning rate.
  --trainDataPath TRAINDATAPATH
                        path to the training data.
  --valDataPath VALDATAPATH
                        path to the validation data.
  --batchSize BATCHSIZE
                        Batch size.
  --epochs EPOCHS       number of epochs.
  --earlyStopping EARLYSTOPPING
                        use early stopping or not.</code></pre>

2. Evaluation and Inference:
<pre><code>usage: test.py [-h] [--mode MODE] [--trainDataPath TRAINDATAPATH] [--valDataPath VALDATAPATH] [--testDataPath TESTDATAPATH] [--modelWeightsPath MODELWEIGHTSPATH] [--batchSize BATCHSIZE]

Test super resolution trained model.

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           evaluate mode or inference mode.
  --trainDataPath TRAINDATAPATH
                        path to the training data.
  --valDataPath VALDATAPATH
                        path to the validation data.
  --testDataPath TESTDATAPATH
                        path to the test data.
  --modelWeightsPath MODELWEIGHTSPATH
                        path to the trained weights.
  --batchSize BATCHSIZE
                        batch size.</code></pre>

# Project structure
<pre><code>├── savepath
│   ├── HR
│   ├── LR
│   └── SR
├── test_data
├── train_data
├── trained
├── data_extraction.py
├── model.py
├── test.py
├── train.py
├── utils.py
├── README.md
├── Pipfile
└── val_data
</code></pre>.
