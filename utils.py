import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np

# Save the training script and all the arguments to a file so that you 
# don't feel like an idiot later when you can't replicate results
import shutil
def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def performance_stats(policies, rewards, matches):

    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)
    accuracy = torch.cat(matches, 0).mean()

    reward = rewards.mean()
    sparsity = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return accuracy, reward, sparsity, variance, policy_set

class LrScheduler:
    def __init__(self, optimizer, base_lr, lr_decay_ratio, epoch_step):
        self.base_lr = base_lr
        self.lr_decay_ratio = lr_decay_ratio
        self.epoch_step = epoch_step
        self.optimizer = optimizer

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (self.lr_decay_ratio ** (epoch // self.epoch_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            if epoch%self.epoch_step==0:
                print '# setting learning_rate to %.2E'%lr


# load model weights trained using scripts from https://github.com/felixgwu/img_classification_pk_pytorch OR
# from torchvision models into our flattened resnets
def load_weights_to_flatresnet(source_model, target_model):

    # compatibility for nn.Modules + checkpoints
    if 'state_dict' not in source_model:
        source_model = {'state_dict': source_model.state_dict()}
    source_state = source_model['state_dict']
    target_state = target_model.state_dict()

    # remove the module. prefix if it exists (thanks nn.DataParallel)
    if source_state.keys()[0].startswith('module.'):
        source_state = {k[7:]:v for k,v in source_state.items()}


    common = set(['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var','fc.weight', 'fc.bias'])
    for key in source_state.keys():

        if key in common:
            target_state[key] = source_state[key]
            continue

        if 'downsample' in key:
            layer, num, item = re.match('layer(\d+).*\.(\d+)\.(.*)', key).groups()
            translated = 'ds.%s.%s.%s'%(int(layer)-1, num, item)
        else:
            layer, item = re.match('layer(\d+)\.(.*)', key).groups()
            translated = 'blocks.%s.%s'%(int(layer)-1, item)


        if translated in target_state.keys():
            target_state[translated] = source_state[key]
        else:
            print translated, 'block missing'

    target_model.load_state_dict(target_state)
    return target_model

def load_checkpoint(rnet, agent, load):
    if load=='nil':
        return None

    checkpoint = torch.load(load)
    if 'resnet' in checkpoint:
        rnet.load_state_dict(checkpoint['resnet'])
        print 'loaded resnet from', os.path.basename(load)
    if 'agent' in checkpoint:
        agent.load_state_dict(checkpoint['agent'])
        print 'loaded agent from', os.path.basename(load)
    # backward compatibility (some old checkpoints)
    if 'net' in checkpoint:
        checkpoint['net'] = {k:v for k,v in checkpoint['net'].items() if 'features.fc' not in k}
        agent.load_state_dict(checkpoint['net'])
        print 'loaded agent from', os.path.basename(load)


def get_transforms(rnet, dset):

    # Only the R32 pretrained model subtracts the mean, sorry :(
    if dset=='C10' and rnet=='R32':
        mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
        std = [x/255.0 for x in [63.0, 62.1, 66.7]]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

    elif dset=='C100' or dset=='C10' and rnet!='R32':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])

    elif dset=='ImgNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])

        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])


    return transform_train, transform_test

# Pick from the datasets available and the hundreds of models we have lying around depending on the requirements.
def get_dataset(model, root='data/'):

    rnet, dset = model.split('_')
    transform_train, transform_test = get_transforms(rnet, dset)

    if dset=='C10':
        trainset = torchdata.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    elif dset=='C100':
        trainset = torchdata.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        testset = torchdata.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    elif dset=='ImgNet':
        trainset = torchdata.ImageFolder(root+'/train/', transform_train)
        testset = torchdata.ImageFolder(root+'/val/', transform_test)

    return trainset, testset

# Make a new if statement for every new model variety you want to index
def get_model(model):

    from models import resnet, base

    if model=='R32_C10':
        rnet_checkpoint = 'cv/pretrained/R32_C10/pk_E_164_A_0.923.t7'
        layer_config = [5, 5, 5]
        rnet = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=10)
        agent = resnet.Policy32([1,1,1], num_blocks=15)

    elif model=='R110_C10':
        rnet_checkpoint = 'cv/pretrained/R110_C10/pk_E_130_A_0.932.t7'
        layer_config = [18, 18, 18]
        rnet = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=10)
        agent = resnet.Policy32([1,1,1], num_blocks=54)

    elif model=='R32_C100':
        rnet_checkpoint = 'cv/pretrained/R32_C100/pk_E_164_A_0.693.t7'
        layer_config = [5, 5, 5]
        rnet = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=100)
        agent = resnet.Policy32([1,1,1], num_blocks=15)

    elif model=='R110_C100':
        rnet_checkpoint = 'cv/pretrained/R110_C100/pk_E_160_A_0.723.t7'
        layer_config = [18, 18, 18]
        rnet = resnet.FlatResNet32(base.BasicBlock, layer_config, num_classes=100)
        agent = resnet.Policy32([1,1,1], num_blocks=54)

    elif model=='R101_ImgNet':
        rnet_checkpoint = 'cv/pretrained/R101_ImgNet/ImageNet_R101_224_76.464'
        layer_config = [3,4,23,3]
        rnet = resnet.FlatResNet224(base.Bottleneck, layer_config, num_classes=1000)
        agent = resnet.Policy224([1,1,1,1], num_blocks=33)

    # load pretrained weights into flat ResNet
    rnet_checkpoint = torch.load(rnet_checkpoint)
    load_weights_to_flatresnet(rnet_checkpoint, rnet)

    return rnet, agent
