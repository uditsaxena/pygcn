from __future__ import division
from __future__ import print_function

import time

import torch.optim as optim
from torch.autograd import Variable

from pygcn.dataset_utils import *
from pygcn.models import *
from pygcn.old_utils import accuracy


def train(model, optimizer, features, adj, labels, idx_train, idx_val, epoch, fastmode=False):
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
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data[0]),
          'acc_train: {:.4f}'.format(acc_train.data[0]),
          'loss_val: {:.4f}'.format(loss_val.data[0]),
          'acc_val: {:.4f}'.format(acc_val.data[0]),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    print("output shape: ", output.shape)
    print("idx test shape: ", idx_test.shape)
    print("labels shape: ", labels.shape)
    # print(labels[idx_test])
    #
    # print(labels[idx_test].shape)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))


def train_cora(args):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data("citeseer")

    # Model and optimizer
    model = Vanilla_GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max() + 1, dropout=args.dropout)

    # print(features.shape[1], args.hidden, labels.max() + 1)
    # model = GCN2_FC1(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max() + 1, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

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
              idx_val=idx_val, epoch=epoch, fastmode=False)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    test(model=model, features=features, adj=adj, labels=labels, idx_test=idx_test)
