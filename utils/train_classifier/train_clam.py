# encoding_size = 1024
# settings = {'num_splits': args.k, 
#             'k_start': args.k_start,
#             'k_end': args.k_end,
#             'task': args.task,
#             'max_epochs': args.max_epochs, 
#             'results_dir': args.results_dir, 
#             'lr': args.lr,
#             'experiment': args.exp_code,
#             'reg': args.reg,
#             'label_frac': args.label_frac,
#             'bag_loss': args.bag_loss,
#             'seed': args.seed,
#             'model_type': args.model_type,
#             'model_size': args.model_size,
#             "use_drop_out": args.drop_out,
#             'weighted_sample': args.weighted_sample,
#             'opt': args.opt}

# if args.model_type in ['clam_sb', 'clam_mb']:
#    settings.update({'bag_weight': args.bag_weight,
#                     'inst_loss': args.inst_loss,
#                     'B': args.B}) 

# TODO
# [ ] adding the Early Stoping
# [ ] adjusting the train_one_epoch
# [ ] adjusting the train 

import torch
import numpy as np
import torch
from utils.utils import *
import os
# from dataset_modules.dataset_generic import save_splits
# from models.model_mil import MIL_fc, MIL_fc_mc
# from models.model_clam import CLAM_MB, CLAM_SB
from src.bag_classifier.clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc




def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error 

def temp_train_loop(features, label, model, optimizer, n_classes, bag_weight, loss_fn=None, device=None):
    logits, Y_prob, Y_hat, _, instance_dict = model(features, label=label, instance_eval=True) 
    loss = loss_fn(logits, label)
    loss_value = loss.item() 
    
    instance_loss = instance_dict['instance_loss']
    # inst_count+=1
    instance_loss_value = instance_loss.item()
    # train_inst_loss += instance_loss_value
    
    total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

    inst_preds = instance_dict['inst_preds']
    inst_labels = instance_dict['inst_labels'] 
    # train_loss += loss_value
    print('loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(loss_value, instance_loss_value, total_loss.item()) + 
        'label: {}, bag_size: {}'.format(label.item(), features.size(0)))

    error = calculate_error(Y_hat, label)
    print("error", error)
    # train_error += error  

# def train_loop_clam(
#     epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, device=None):
#     model.train()
    
#     train_loss = 0.
#     train_error = 0.
#     train_inst_loss = 0.
#     inst_count = 0

#     print('\n')
#     for batch_idx, (data, label) in enumerate(loader):
#         data, label = data.to(device), label.to(device)
#         logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        
#         loss = loss_fn(logits, label)
#         loss_value = loss.item()

#         instance_loss = instance_dict['instance_loss']
#         inst_count+=1
#         instance_loss_value = instance_loss.item()
#         train_inst_loss += instance_loss_value
        
#         total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

#         inst_preds = instance_dict['inst_preds']
#         inst_labels = instance_dict['inst_labels']

#         train_loss += loss_value
#         if (batch_idx + 1) % 20 == 0:
#             print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
#                 'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

#         error = calculate_error(Y_hat, label)
#         train_error += error
        
#         # backward pass
#         total_loss.backward()
#         # step
#         optimizer.step()
#         optimizer.zero_grad()


#     # calculate loss and error for epoch
#     train_loss /= len(loader)
#     train_error /= len(loader)
    
#     # if inst_count > 0:
#     #     train_inst_loss /= inst_count
#     #     print('\n')
#     #     for i in range(2):
#     #         acc, correct, count = inst_logger.get_summary(i)
#     #         print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

#     print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
#     # for i in range(n_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    #     if writer and acc is not None:
    #         writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    # if writer:
    #     writer.add_scalar('train/loss', train_loss, epoch)
    #     writer.add_scalar('train/error', train_error, epoch)
    #     writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, logger = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    # Log header for the epoch
    logger.info(f"Starting epoch {epoch}...")

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count += 1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1 - bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value

        if (batch_idx + 1) % 20 == 0:
            # Log information every 20 batches
            logger.info(f"Batch {batch_idx}, Loss: {loss_value:.4f}, Instance Loss: {instance_loss_value:.4f}, "
                        f"Weighted Loss: {total_loss.item():.4f}, Label: {label.item()}, Bag Size: {data.size(0)}")

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # Backward pass and optimizer step
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate and log the average loss and error for the epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        # Log instance accuracy for each class
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            logger.info(f"Class {i} Clustering Accuracy: {acc}, Correct: {correct}/{count}")

    logger.info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Clustering Loss: {train_inst_loss:.4f}, Train Error: {train_error:.4f}")

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        logger.info(f"Class {i}: Accuracy: {acc}, Correct: {correct}/{count}")

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch) 

if __name__=='__main__':
    logger = setup_logger("./logs/training_log.txt")
    epoch = 0
    model = CLAM_SB()
    train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, logger, loss_fn)
 
 

