from __future__ import division
from __future__ import print_function

import time

import torch.optim as optim
from torch.autograd import Variable

from pygcn.dataset_utils import *
from pygcn.models import *
from pygcn.old_utils import accuracy


def train(model, optimizer, features, adj, labels, idx_train, idx_val, epoch, file="a.csv", fastmode=False):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    epoch_str = 'Epoch: {:04d}'.format(epoch + 1)
    train_loss_str = 'loss_train: {:.4f}'.format(loss_train.data[0])
    train_acc_str = 'acc_train: {:.4f}'.format(acc_train.data[0])
    val_loss_str = 'loss_val: {:.4f}'.format(loss_val.data[0])
    val_acc_str = 'acc_val: {:.4f}'.format(acc_val.data[0])
    time_str = 'time: {:.4f}s'.format(time.time() - t)

    print(epoch_str, train_loss_str, train_acc_str, val_loss_str, val_acc_str, time_str)

    with open(file, 'a+') as f:
        f.write(
            epoch_str + ", " + train_loss_str + ", " + train_acc_str + ", " + val_loss_str + ", " + val_acc_str + "\n")


def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))


def train_cora(args):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data("cora")

    # Model and optimizer
    # if args.model_type == 0:
    model = Vanilla_GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max() + 1, dropout=args.dropout)
    if args.model_type == 1:
        model = GCN2_FC1(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max() + 1, dropout=args.dropout)
    if args.model_type == 2:
        model = GCN2_LC_NL_FC1(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max() + 1, dropout=args.dropout)
    if args.model_type == 3:
        model = GCN2_LC_L_FC1(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max() + 1, dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    filename = args.dataset + "_" + str(args.model_type) + "_" + str(args.lr) + "_" + str(args.hidden) + "_" + str(
        args.dropout)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    features, labels = Variable(features), Variable(labels)

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(model=model, optimizer=optimizer, features=features, adj=adj, labels=labels, idx_train=idx_train,
              idx_val=idx_val, epoch=epoch, fastmode=False, file=filename)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    test(model=model, features=features, adj=adj, labels=labels, idx_test=idx_test)
