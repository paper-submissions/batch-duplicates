# "Increasing batch size through instance repetition improves generalization" - PyTorch code

This is a complete training example for Deep Convolutional Networks on various datasets (ImageNet, Cifar10, Cifar100, MNIST).

It is based off [imagenet example in pytorch](https://github.com/pytorch/examples/tree/master/imagenet) with helpful additions such as:
  - Training on several datasets other than imagenet
  - Complete logging of trained experiment
  - Graph visualization of the training/validation loss and accuracy
  - Definition of preprocessing and optimization regime for each model
  - Distributed training
 
 To clone:
 ```
 git clone --recursive https://github.com/eladhoffer/convNet.pytorch
 ```

This code can be used to implement the paper "Increasing batch size through instance repetition improves generalization"
For example, training the resnet44 + cutout example in paper:
 ```
 python main.py --dataset cifar10 --model resnet --model-config "{'depth': 44}"  --duplicates 40 --cutout -b 64 --epochs 100 --save resnet44_cutout_m-40
 ```
    
## Dependencies

- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization


## Data
- Configure your dataset path with ``datasets-dir`` argument
- To get the ILSVRC data, you should register on their site for access: <http://www.image-net.org/>
