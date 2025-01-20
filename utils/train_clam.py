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

import torch


# train function for CLAM