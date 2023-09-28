

# CPKD-PyTorch
The official pytorch implementation of **CPKD: Channel and Position-wise Knowledge Distillation for Segmentation of Negative Obstacles**. 

We test our code in Python 3.7, CUDA 11.1, cuDNN 8, and PyTorch 1.7.1. We provide `Dockerfile` to build the docker image we used. You can modify the `Dockerfile` as you want.  
<div align=center>
<img src="docs/overall.png" width="900px"/>
</div>


# Introduction
CPKD is a knowledge distillation framework for the segmentation of positive obstacles which can transfer knowledge between two feature maps with different resolutions and channel numbers

# Dataset
The [NPO dataset](https://github.com/lab-sun/InconSeg/blob/main/docs/dataset.md) can be downloaded from here [page](https://labsun-me.polyu.edu.hk/zfeng/InconSeg/). You can also download the dataset from [Baidu Netdisk](https://pan.baidu.com/s/1oxUb-0vdiZzTPu4waci39g), the password is cekd






# Acknowledgement
Some of the codes are borrowed from [RTFNet](https://github.com/yuxiangsun/RTFNet) 
