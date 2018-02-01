import os
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
from torch.distributions import Bernoulli

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


import argparse
parser = argparse.ArgumentParser(description='BlockDrop Training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--model', default='R110_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load rnet+agent from')
parser.add_argument('--pretrained', default=None, help='pretrained policy model checkpoint (from curriculum training)')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=1600, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=2000, help='total epochs to run')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, help='lr *= lr_decay_ratio after epoch_steps')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
# parser.add_argument('--joint', action ='store_true', default=True, help='train both the policy network and the resnet')
parser.add_argument('--penalty', type=float, default=-5, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def get_reward(preds, targets, policy):

    block_use = policy.sum(1).float()/policy.size(1)
    sparse_reward = 1.0-block_use**2

    _, pred_idx = preds.max(1)
    match = (pred_idx==targets).data

    reward = sparse_reward
    reward[1-match] = args.penalty
    reward = reward.unsqueeze(1)

    return reward, match.float()


def train(epoch):

    agent.train()
    rnet.train()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        probs, value = agent(inputs)

        #---------------------------------------------------------------------#

        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        probs = probs*args.alpha + (1-probs)*(1-args.alpha)
        distr = Bernoulli(probs)
        policy = distr.sample()

        v_inputs = Variable(inputs.data, volatile=True)
        preds_map = rnet.forward(v_inputs, policy_map)
        preds_sample = rnet.forward(inputs, policy)

        reward_map, _ = get_reward(preds_map, targets, policy_map.data)
        reward_sample, match = get_reward(preds_sample, targets, policy.data)

        advantage = reward_sample - reward_map
        # advantage = advantage.expand_as(policy)
        loss = -distr.log_prob(policy).sum(1, keepdim=True) * Variable(advantage)
        loss = loss.sum()

        #---------------------------------------------------------------------#
        loss += F.cross_entropy(preds_sample, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        matches.append(match.cpu())
        rewards.append(reward_sample.cpu())
        policies.append(policy.data.cpu())

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    log_str = 'E: %d | A: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d'%(epoch, accuracy, reward, sparsity, variance, len(policy_set))
    print log_str

    log_value('train_accuracy', accuracy, epoch)
    log_value('train_reward', reward, epoch)
    log_value('train_sparsity', sparsity, epoch)
    log_value('train_variance', variance, epoch)
    log_value('train_unique_policies', len(policy_set), epoch)


def test(epoch):

    agent.eval()
    rnet.eval()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        probs, _ = agent(inputs)

        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0
        policy = Variable(policy)

        preds = rnet.forward(inputs, policy)
        reward, match = get_reward(preds, targets, policy.data)

        matches.append(match)
        rewards.append(reward)
        policies.append(policy.data)

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    log_str = 'TS - A: %.3f | R: %.2E | S: %.3f | V: %.3f | #: %d'%(accuracy, reward, sparsity, variance, len(policy_set))
    print log_str

    log_value('test_accuracy', accuracy, epoch)
    log_value('test_reward', reward, epoch)
    log_value('test_sparsity', sparsity, epoch)
    log_value('test_variance', variance, epoch)
    log_value('test_unique_policies', len(policy_set), epoch)

    # save the model
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()

    state = {
      'agent': agent_state_dict,
      'resnet': rnet_state_dict,
      'epoch': epoch,
      'reward': reward,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_R_%.2E_S_%.2f_#_%d.t7'%(epoch, accuracy, reward, sparsity, len(policy_set)))


#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
rnet, agent = utils.get_model(args.model)

if args.pretrained is not None:
    checkpoint = torch.load(args.pretrained)
    key = 'net' if 'net' in checkpoint else 'agent'
    agent.load_state_dict(checkpoint[key])
    print 'loaded pretrained model from', args.pretrained

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    rnet.load_state_dict(checkpoint['resnet'])
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print 'loaded agent from', args.load


if args.parallel:
    agent = nn.DataParallel(agent)
    rnet = nn.DataParallel(rnet)

rnet.cuda()
agent.cuda()

optimizer = optim.Adam(list(agent.parameters())+list(rnet.parameters()), lr=args.lr, weight_decay=args.wd)

configure(args.cv_dir+'/log', flush_secs=5)
lr_scheduler = utils.LrScheduler(optimizer, args.lr, args.lr_decay_ratio, args.epoch_step)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    lr_scheduler.adjust_learning_rate(epoch)

    train(epoch)
    if epoch%10==0:
        test(epoch)
