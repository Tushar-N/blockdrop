# BlockDrop: Dynamic Inference Paths in Residual Networks
![BlockDrop Model](https://user-images.githubusercontent.com/4995097/35775877-3cc64f86-0957-11e8-85c4-9bd16cda22a0.png)

This code implements a policy network that learns to dynamically choose which blocks of a ResNet to execute during inference so as to best reduce total computation without degrading prediction accuracy. Built upon a ResNet-101 model, our method achieves a speedup of 20% on average, going as high as 36% for some images, while maintaining the same 76.4% top-1 accuracy on ImageNet.

This is the code accompanying the work:  
Zuxuan Wu*, Tushar Nagarajan*, Abhishek Kumar, Steven Rennie, Larry S. Davis, Kristen Grauman, and Rogerio Feris. BlockDrop: Dynamic Inference Paths in Residual Networks [[arxiv]](https://arxiv.org/pdf/1711.08393.pdf)  
(* authors contributed equally)

## Prerequisites
The code is written and tested using Python (2.7) and PyTorch (v0.3.0).

**Packages**: Install using `pip install -r requirements.txt`

**Pretrained models**: Our models require standard pretrained ResNets on CIFAR and ImageNet as starting points. These can be trained using [this](https://github.com/felixgwu/img_classification_pk_pytorch) repository, or can be obtained directly from us

```bash
wget -O blockdrop-checkpoints.tar.gz https://www.cs.utexas.edu/~tushar/blockdrop/blockdrop-checkpoints.tar.gz
tar -zxvf blockdrop-checkpoints.tar.gz
```
The downloaded checkpoints will be unpacked to `./cv/` for further use. The folder also contains various checkpoints from each stage of training.

**Datasets**: PyTorch's *torchvision* package automatically downloads CIFAR10 and CIFAR100 during training. ImageNet must be downloaded and organized following [these steps](https://github.com/soumith/imagenet-multiGPU.torch#data-processing).

## Training a model
Training occurs in two steps (1) Curriculum Learning and (2) Joint Finetuning.  
Models operating on ResNets of different depths can be trained on different datasets using the same script. Examples of how to train these models are given below. Checkpoints and tensorboard log files will be saved to folder specified in `--cv_dir`

#### Curriculum Learning
The policy network can be trained using a CL schedule as follows.

```bash
# Train a model on CIFAR 10 built upon a ResNet-110
python cl_training.py --model R110_C10 --cv_dir cv/R110_C10_cl/ --lr 1e-3 --batch_size 2048 --max_epochs 5000

# Train a model on ImageNet built upon a ResNet-101
python cl_training.py --model R101_ImgNet --cv_dir cv/R101_ImgNet_cl/ --lr 1e-3 --batch_size 2048 --max_epochs 45 --data_dir data/imagenet/
```

Model checkpoints after the curriculum learning step can be found in the downloaded folder. For example: `./cv/cl_learning/R110_C10/ckpt_E_5300_A_0.754_R_2.22E-01_S_20.10_#_7787.t7`

#### Joint Finetuning
Checkpoints trained during the curriculum learning phase can be used to further jointly finetune the base ResNet to achieve the results reported in the paper. Different values for the penalty parameter control the trade-off between accuracy and speed.

```bash
# Finetune a ResNet-110 on CIFAR 10 using the checkpoint from cl_training
python finetune.py --model R110_C10 --lr 1e-4 --penalty -10 --pretrained cv/cl_training/R110_C10/ckpt_E_5300_A_0.754_R_2.22E-01_S_20.10_#_7787.t7 --batch_size 256  --max_epochs 2000 --cv_dir cv/R110_C10_ft_-10/

# Finetune a ResNet-101 on ImageNet using the checkpoint from cl_training
python finetune.py --model R101_ImgNet --lr 1e-4  --penalty -5 --pretrained cv/cl_training/R101_ImgNet/ckpt_E_4_A_0.746_R_-3.70E-01_S_29.79_#_484.t7 --data_dir data/imagenet/ --batch_size 320 --max_epochs 10 --cv_dir cv/R101_ImgNet_ft_-5/
```

Model checkpoints after the joint finetuning step can be found in the downloaded folder. For example: `./cv/finetuned/R101_ImgNet_gamma_5/ckpt_E_10_A_0.764_R_-8.46E-01_S_24.77_#_10.t7`

## Testing and Profiling
Once jointly finetuned, models can be profiled for accuracy and FLOPs counts.
```bash
python test.py --model R110_C10 --load cv/finetuned/R110_C10_gamma_10/ckpt_E_2000_A_0.936_R_1.95E-01_S_16.93_#_469.t7
```
The model should produce an accuracy of 93.6% and use 1.81E+08 FLOPs on average. The output should look like this:
```
    Accuracy: 0.936
    Block Usage: 16.933 ± 3.717
    FLOPs/img: 1.81E+08 ± 3.43E+07
    Unique Policies: 469
```

The ImageNet model can be evaluated in a similar manner, and will generate a corresponding output.
```
python test.py --model R101_ImgNet --load cv/finetuned/R101_ImgNet_gamma_5/ckpt_E_10_A_0.764_R_-8.46E-01_S_24.77_#_10.t7
```
```
    Accuracy: 0.764
    Block Usage: 24.770 ± 0.980
    FLOPs/img: 1.25E+10 ± 4.28E+08
    Unique Policies: 10
```


## Visualization
Learned policies over ResNet blocks show that there is a clear separation between easy/hard images in terms of the number of blocks they require. In addition, unique policies over the blocks admit distinct image styles.

![Policy visualization](https://user-images.githubusercontent.com/4995097/35775878-3e5ee4e8-0957-11e8-832d-b9dc2ea8fecc.png)

For more qualitative results, see Sec. 4.3 and Figures 4. and 5. in the paper.



## Cite

If you find this repository useful in your own research, please consider citing:
```
@inproceedings{blockdrop,
  title={BlockDrop: Dynamic Inference Paths in Residual Networks},
  author={Wu, Zuxuan and Nagarajan, Tushar and Kumar, Abhishek and Rennie, Steven and Davis, Larry S and Grauman, Kristen and Feris, Rogerio},
  booktitle={CVPR},
  year={2018}
}
```
