import os
import time
import random
from args import args

import torch
import torch.nn.functional as F
import torch.optim as optim

from data.data_loader import load_data
from utils import accuracy, seed_everything
from gat import GraphAttnNetwork


def train_one_epoch(features, adj, labels, idx_train, idx_val, model, epoch, optimizer):
    t = time.time()
    # training mode
    model.train()
    # zero gradients
    optimizer.zero_grad()
    # list of predictions
    output_list = model(features, adj)
    # obtain averaged loss of all prediction attentions
    loss_train = torch.stack([F.nll_loss(output[idx_train], labels[idx_train]) for output in output_list], dim=0).mean(dim=0)
    # obtain averaged predictions
    acc_train = accuracy(torch.stack([output[idx_train] for output in output_list], dim=0).mean(dim=0), labels[idx_train])
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch),
          'Loss_train: {:.4f}'.format(loss_train.data.item()),
          'Acc_train: {:.4f}'.format(acc_train.data.item()),
          'Time: {:.4f}s'.format(time.time() - t))
    return


def val_one_epoch(features, adj, labels, idx_val, model):
    # testing mode
    model.eval()

    output_list = model(features, adj)
    loss_val = torch.stack([F.nll_loss(output[idx_val], labels[idx_val]) for output in output_list], dim=0).mean(dim=0)
    acc_val = accuracy(torch.stack([output[idx_val] for output in output_list], dim=0).mean(dim=0), labels[idx_val])
    return loss_val.data.item(), acc_val.data.item()


def final_test(features, adj, labels, idx_test, model):
    # testing mode
    model.eval()

    output_list = model(features, adj)
    loss_test = torch.stack([F.nll_loss(output[idx_test], labels[idx_test]) for output in output_list], dim=0).mean(dim=0)
    acc_test = accuracy(torch.stack([output[idx_test] for output in output_list], dim=0).mean(dim=0), labels[idx_test])
    return loss_test.data.item(), acc_test.data.item()


def main(args):
    # Load train/val/test data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # Model construction
    model = GraphAttnNetwork(in_dim=features.shape[1],
                c_dim=args.c_dim,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                num_gals_in=args.num_gals_in,
                num_gals_predict=args.num_gals_predict,
                alpha=args.alpha)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    # Start training model
    t_total = time.time()
    loss_vals = []
    acc_vals = []
    epoch_vals = []
    best_acc = -1
    best_epoch = -1
    for epoch in range(args.epochs):
        train_one_epoch(features, adj, labels, idx_train, idx_val, model, epoch, optimizer)
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            loss_val, acc_val = val_one_epoch(features, adj, labels, idx_val, model)
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            epoch_vals.append(epoch)

            if acc_val > best_acc:
                if best_epoch != -1:
                    os.remove('model_{}.pth'.format(best_epoch))
                best_acc = acc_val
                best_epoch = epoch
                torch.save(model.state_dict(), 'model_{}.pth'.format(best_epoch))

    print("Complete training")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('*********The validation accuracy*********')
    for epoch, loss, acc in zip(epoch_vals, loss_vals, acc_vals):
        print(f'Epoch: {epoch:3d}   Loss: {loss:.4f}    Acc: {acc:.4f}')
    print('*****************************************')

    # Load the saved best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('model_{}.pth'.format(best_epoch)))

    # Final testing using the best model
    loss_final, acc_final = final_test(features, adj, labels, idx_test, model)
    print(f'After training GAT model for {args.epochs} epochs, the final test results are:')
    print(f"Loss: {loss_final:.4f}, Acc: {acc_final:.4f}.")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    assert torch.cuda.is_available()
    seed_everything(args.seed)
    main(args)