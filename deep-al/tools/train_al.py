# Code is originally from the Typiclust (https://arxiv.org/abs/2202.02794) and ProbCover (https://arxiv.org/abs/2205.11320) implementation
# from https://github.com/avihu111/TypiClust by Avihu Dekel and Guy Hacohen which is licensed under MIT license
# You may obtain a copy of the License at
#
# https://github.com/avihu111/TypiClust/blob/main/LICENSE
#
####################################################################################

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from copy import deepcopy

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.core.builders import NNNet, FeaturesNet, KernelizedNet, LinearKernel
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter
from pycls.utils.al import BudgetIterator
from pycls.calibration.metrics import ECELoss
from pycls.calibration.temperature_scaling import obtain_temperature
from pycls.models.deit import VisionTransformer

logger = lu.get_logger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--ent_weight', default=0.0, help='entropy weight', type=float)
    parser.add_argument('--lr', default=-1.0, help='learning rate', type=float)
    parser.add_argument('--exp-name', help='Experiment Name', required=True, type=str)
    parser.add_argument('--tag', help='tag', default='', type=str)
    parser.add_argument('--budget', help='Budget Per Round', required=True)
    parser.add_argument('--max_iter', help='maximum iter', type=int, default=10)
    parser.add_argument('--initial_size', help='Size of the initial random labeled set', default=0, type=int)
    parser.add_argument('--seed', help='Random seed', default=1, type=int)
    parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False)
    parser.add_argument('--linear_from_features', help='Whether to use a linear layer from self-supervised features', action='store_true')

    parser.add_argument('--kernelize', help='Whether to use a kernelized layer from self-supervised features', action='store_true')
    parser.add_argument('--nn', help='Whether to use nearest neighbor classifier', action='store_true')

    # Active learning
    parser.add_argument('--al', help='AL Method', required=True, type=str)
    parser.add_argument('--delta', help='Relevant only for ProbCover', default=0.6, type=float)
    parser.add_argument('--kernel', help='kernel name for herding', default=None, type=str)
    parser.add_argument('--feature', help='selection features', default='simclr', type=str)
    parser.add_argument('--unc_feature', help='selection features', default='simclr', type=str)
    parser.add_argument('--herding_init', help='init with herding results for kkmeoids',
                        type=str2bool, default=False)
    parser.add_argument('--unc_trans_fn', help='function to transform uncertainty',
                        default='identity', type=str)
    parser.add_argument('--last_layer', help='type of last layer',
                        default='fc', type=str)
    parser.add_argument('--normalize', help='normalization for uherding', type=str2bool, default=False)
    parser.add_argument('--adaptive_delta', help='use adaptive_delta', type=str2bool, default=False)

    # Calibration
    parser.add_argument('--gamma', help='gamma for focal loss', type=float, default=0)
    parser.add_argument('--temp_scale', help='use temperatue sacling', type=str2bool, default=False)
    return parser


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def main(cfg):
    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    # set random seed
    random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.RNG_SEED)

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)

    # Create "DATASET/MODEL TYPE" specific directory
    model_type_short = str(cfg.MODEL.TYPE.split('_')[0])
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, model_type_short)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory
    # all logs, labeled, unlabeled, validation sets are stroed here
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}

    cfg.TAG = args.tag

    ufeature = f"{args.unc_feature}"
    cfg.ACTIVE_LEARNING.UNC_FEATURE = ufeature
    assert ufeature in ["classifier", "simclr", "scan", "dino"]

    base_lr = cfg.OPTIM.BASE_LR
    if args.nn:
        model_feature = f"{ufeature}_nn"
    elif args.linear_from_features:
        model_feature = f"{ufeature}"
    elif args.feature == 'finetune':
        model_feature = 'finetune'
        cfg.MODEL.FINETUNE = True
        cfg.ACTIVE_LEARNING.FEATURE = 'classifier'
        assert cfg.MODEL.CHECKPOINT_PATH, f'For finetune, checkpoint path should exist. It is now empty'
    else:
        model_feature = "random"
        cfg.ACTIVE_LEARNING.FEATURE = 'classifier'

    if cfg.EXP_NAME == 'auto':
        train_params = f'loss_{args.loss}_entw_{args.ent_weight}_lr_{cfg.OPTIM.BASE_LR}_ly_{args.last_layer}'
        feature_params = f'feature_{model_feature}_ufeature_{ufeature}'
        al_params = f'delta_{args.delta}_kernel_{args.kernel}_budget_{args.budget}'
        uherding_params = f'norm_{args.normalize}_adapt_{args.adaptive_delta}_ts_{args.temp_scale}'

        exp_dir = f'{args.al}_{train_params}_{feature_params}_{al_params}_{uherding_params}_seed_{args.seed}'
        if args.tag:
            exp_dir = f'{exp_dir}_tag_{args.tag}'
    else:
        exp_dir = cfg.EXP_NAME

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    cfg.ACTIVE_LEARNING.INIT_L_RATIO = args.initial_size / train_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO, \
        val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)

    # Initialize the model.
    model = model_builder.build_model(cfg).cuda()
    print("model: {}\n".format(cfg.MODEL.TYPE))
    logger.info("model: {}\n".format(cfg.MODEL.TYPE))
    print(model)

    budget_iterator = BudgetIterator(cfg, num_train=len(train_data), init_num=len(lSet))
    budget_list = budget_iterator.budget_list

    if args.feature == 'finetune':
        lr_list = np.linspace(base_lr / 1.0, base_lr / 1.0, cfg.ACTIVE_LEARNING.MAX_ITER+1)
        cfg.OPTIM.BASE_LR = lr_list[0].item()
    else:
        lr_list = np.linspace(base_lr, base_lr, cfg.ACTIVE_LEARNING.MAX_ITER+1)
    print(f'lr list: {lr_list}')

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    ## Preparing dataloaders for initial training
    lSet_loader = None
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TEST.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    test_data.no_aug = True

    # Construct the optimizer
    if not cfg.MODEL.KERNELIZE and not cfg.MODEL.NN:
        optimizer = optim.construct_optimizer(cfg, model)
        opt_init_state = deepcopy(optimizer.state_dict())
        model_init_state = deepcopy(model.state_dict().copy())

        logger.info("optimizer: {}\n".format(optimizer))

    print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))

    checkpoint_file = None
    sample_times = []
    test_accs = []
    test_eces = []
    test_ents = []
    temps = []
    deltas = []
    budget_regimes = []
    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir
        cfg.COVERAGE_PATH = os.path.join(episode_dir, 'coverage.npy')
        cfg.UNCERTAINTY_PATH = os.path.join(episode_dir, 'uncertainty.npy')
        cfg.INDICES_PATH = os.path.join(episode_dir, 'indices.npy')
        cfg.LOGIT_PATH = os.path.join(episode_dir, 'logits.npy')

        # Active Sample
        print("======== ACTIVE SAMPLING ========\n")
        logger.info("======== ACTIVE SAMPLING ========\n")

        budget = next(budget_iterator)
        if budget is None: break

        al_condition = (cur_episode == 0 and len(lSet) == 0) or (cur_episode > 0)
        if al_condition:
            if len(lSet) > 0 and args.temp_scale:
                temp_clf_model = model_builder.build_model(cfg)
                temp_clf_model.load_state_dict(deepcopy(model_init_state))
                temp_optimizer = optim.construct_optimizer(cfg, temp_clf_model)

                num_temp_vSet = int(len(lSet) * 0.3)

                temp_lSet, temp_valSet = lSet[:-num_temp_vSet], lSet[-num_temp_vSet:]
                temp_lSet_loader = data_obj.getIndexesDataLoader(indexes=temp_lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
                temp_valSet_loader = data_obj.getIndexesDataLoader(indexes=temp_valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)

                best_val_acc, best_val_epoch, tmp_checkpoint_file = train_model(
                    temp_lSet_loader, temp_valSet_loader, temp_clf_model, temp_optimizer, cfg)
                temp = obtain_temperature(
                    cfg, temp_clf_model, temp_valSet_loader, temp_values=None, verbose=True, temps=temps)
                temp_additiona_scale = 1.0
                temp = temp * temp_additiona_scale
                print(f'temp additional scaling: {temp_additiona_scale}')
                cfg.ACTIVE_LEARNING.TEMP = float(temp.item())
                print(f'moving temperature: {temp}')

                del temp_clf_model
                if cur_episode == 0:
                   model = model_builder.build_model(cfg)

            if cur_episode > 0:
                clf_model = model_builder.build_model(cfg)
                clf_model = cu.load_checkpoint(checkpoint_file, clf_model)
            else:
                print('Labeled Set is Empty - Sampling an Initial Pool')
                clf_model = model

            al_obj = ActiveLearning(data_obj, cfg, budget=budget)
            start_time = time.time()
            activeSet, new_uSet = al_obj.sample_from_uSet(
                clf_model, lSet, uSet, train_data, val_dataloader=valSet_loader,
                train_dataloader=lSet_loader)
            sample_time = time.time() - start_time
            sample_times.append(sample_time)
            if hasattr(al_obj, 'delta'):
                deltas.append(al_obj.delta)
            if hasattr(al_obj, 'budget_regime'):
                budget_regimes.append(al_obj.budget_regime)

            print(f'temperatues: {temps}')
            print(f'deltas: {deltas}')
            print(f'budget_regimes: {budget_regimes}')
            print(f'sample_times: {sample_times}')

            # Save current lSet, new_uSet and activeSet in the episode directory
            data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)

            # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
            lSet = np.append(lSet, activeSet)
            uSet = new_uSet

            lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
            uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)

            print("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
            logger.info("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
            print("================================\n\n")
            logger.info("================================\n\n")
        else:
            lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
            print(f"Active Sampling Skippped - Episode: {cur_episode}, lSet: {len(lSet)}")

        if checkpoint_file is not None:
            os.remove(checkpoint_file)

        # Train model
        print("======== TRAINING ========")
        logger.info("======== TRAINING ========")

        if not cfg.MODEL.KERNELIZE and not cfg.MODEL.NN:
            best_val_acc, best_val_epoch, checkpoint_file = train_model(
                lSet_loader,
                valSet_loader,
                model, optimizer, cfg)

            print("Best Validation Accuracy: {}\nBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))
            logger.info("EPISODE {} Best Validation Accuracy: {}\tBest Epoch: {}\n".format(cur_episode, round(best_val_acc, 4), best_val_epoch))
        else:
            checkpoint_file = None

        # Test best model checkpoint
        print("======== TESTING ========\n")
        logger.info("======== TESTING ========\n")
        train_loader = data_obj.getSequentialDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        test_acc, test_ece, test_ent = test_model(test_loader, checkpoint_file, cfg, cur_episode, train_loader)
        print("Test Accuracy: {} Test ECE: {}.\n".format(round(test_acc, 4), round(test_ece, 4)))
        logger.info("EPISODE {} Test Accuracy {}, Test ECE {}.\n".format(cur_episode, test_acc, test_ece))
        test_accs.append(test_acc)
        test_eces.append(test_ece)
        test_ents.append(test_ent)
        print(test_accs)
        print(test_eces)
        print(test_ents)

        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        if not cfg.MODEL.KERNELIZE and not cfg.MODEL.NN:
            # start model from scratch
            print('Starting model from scratch - ignoring existing weights.')
            model = model_builder.build_model(cfg)
            model.load_state_dict(deepcopy(model_init_state))

            if args.feature == 'finetune' and len(lr_list) > cur_episode+1:
                cfg.OPTIM.BASE_LR = lr_list[cur_episode+1].item()

            # Construct the optimizer
            optimizer = optim.construct_optimizer(cfg, model)

    if checkpoint_file is not None:
        os.remove(checkpoint_file)

    test_accs = np.array(test_accs)
    acc_path = os.path.join(cfg.EXP_DIR, 'test_accs.npy')
    np.save(acc_path, test_accs)
    print(f'save acc_path to {acc_path}')

    test_eces = np.array(test_eces)
    ece_path = os.path.join(cfg.EXP_DIR, 'test_eces.npy')
    np.save(ece_path, test_eces)
    print(f'save ece_path to {ece_path}')

    test_ents = np.array(test_ents)
    ent_path = os.path.join(cfg.EXP_DIR, 'test_ents.npy')
    np.save(ent_path, test_ents)
    print(f'save ent_path to {ent_path}')

    sample_times = np.array(sample_times)
    time_path = os.path.join(cfg.EXP_DIR, 'sample_times.npy')
    np.save(time_path, sample_times)
    print(f'save sample_times to {time_path}')

    temps = np.array(temps)
    temp_path = os.path.join(cfg.EXP_DIR, 'temps.npy')
    np.save(temp_path, temps)
    print(f'save temps to {temp_path}')

    deltas = np.array(deltas)
    delta_path = os.path.join(cfg.EXP_DIR, 'deltas.npy')
    np.save(delta_path, deltas)
    print(f'save deltas to {delta_path}')



def train_model(train_loader, val_loader, model, optimizer, cfg):
    start_epoch = 0
    loss_fun = losses.get_loss_fun(cfg)

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))

    # Perform the training loop
    logger.info('Start epoch: {}'.format(start_epoch + 1))
    val_set_acc = 0.
    val_set_ece = np.inf

    temp_best_val_acc = 0.
    temp_best_val_epoch = 0

    # Best checkpoint model and optimizer states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []
    val_ece_epochs = []

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader)/cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25
    clf_iter_count = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_loss, clf_iter_count = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, \
                                        cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_train_iterations)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)

        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_loader.dataset.no_aug = True
            val_set_err, val_set_ece, val_set_ent = test_epoch(val_loader, model, val_meter, cur_epoch,
                                                  train_loader=train_loader, loss_fun=loss_fun)
            val_set_acc = 100. - val_set_err
            val_loader.dataset.no_aug = False
            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()

                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            # Since we start from 0 epoch
            val_acc_epochs_x.append(cur_epoch+1)
            val_acc_epochs_y.append(val_set_acc)
            val_ece_epochs.append(val_set_ece)

        logger.info("Successfully logged numpy arrays!!")

        print('Training Epoch: {}/{}\tTrain Loss: {}\tVal Accuracy: {}\tVal ECE:{}'.format(
            cur_epoch+1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4),
            round(val_set_acc, 4), round(val_set_ece, 4)))

    # Save the best model checkpoint (Episode level)
    checkpoint_file = cu.save_checkpoint(info="vlBest_acc_"+str(int(temp_best_val_acc)), \
        model_state=best_model_state, optimizer_state=best_opt_state, epoch=temp_best_val_epoch, cfg=cfg)

    print('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
    logger.info('Wrote Best Model Checkpoint to: {}\n'.format(checkpoint_file))

    best_val_acc = temp_best_val_acc
    best_val_epoch = temp_best_val_epoch

    return best_val_acc, best_val_epoch, checkpoint_file


def test_model(test_loader, checkpoint_file, cfg, cur_episode, train_loader=None, loss_fun=None):
    test_meter = TestMeter(len(test_loader))

    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)
    print(f'Loaded a checkpoint from: {checkpoint_file}')

    test_err, test_ece, test_ent = test_epoch(test_loader, model, test_meter, cur_episode,
                                    train_loader=train_loader, loss_fun=loss_fun, is_val=False)
    test_acc = 100. - test_err

    return test_acc, test_ece, test_ent


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""
    assert not train_loader.dataset.no_aug
    # Shuffle the data
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    for cur_iter, (inputs, labels) in enumerate(train_loader):
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        output_dict = model(inputs, labels)

        preds = output_dict['preds']
        labels = output_dict['labels']
        if isinstance(loss_fun, torch.nn.MSELoss):
            onehot_labels = output_dict['onehot_labels']
            loss = loss_fun(preds, onehot_labels)
        else:
            loss = loss_fun(preds, labels)

        # Compute the loss
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()

        if isinstance(model, VisionTransformer):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])

        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()

        #Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return loss, clf_iter_count


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch, train_loader=None, loss_fun=None,
               is_val=True):
    """Evaluates the model on the test set."""
    assert test_loader.dataset.no_aug
    use_softmax = not (isinstance(loss_fun, (torch.nn.MSELoss, losses.RescaleMSELoss)) or loss_fun is None)
    ece_meter = ECELoss(num_classes=model.num_classes, use_softmax=use_softmax)

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.
    if train_loader is not None and (cfg.MODEL.KERNELIZE or cfg.MODEL.NN):
        total_train_inputs = []
        total_train_labels = []

        for train_inputs, train_labels in train_loader:
            total_train_inputs.append(train_inputs)
            total_train_labels.append(train_labels)

        total_train_inputs = torch.cat(total_train_inputs).cuda()
        total_train_labels = torch.cat(total_train_labels).cuda()

    test_logits = []
    test_labels = []
    test_entropies = []
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)

            # Compute the predictions
            if cfg.MODEL.KERNELIZE or cfg.MODEL.NN:
                output_dict = model(total_train_inputs, total_train_labels, inputs)
            else:
                output_dict = model(inputs, labels)

            preds = output_dict['preds']
            if 'labels' in output_dict:
                labels = output_dict['labels']

            probs = torch.nn.functional.softmax(preds, dim=1)
            entropy = probs * torch.log(probs + 1e-8)
            entropy = -1*torch.sum(entropy, dim=1)
            test_entropies.append(entropy.detach().cpu().numpy())

            # Compute the errors
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])

            # Copy the errors from GPU to CPU (sync point)
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0)*cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()

            test_logits.append(preds)
            test_labels.append(labels)

    if test_entropies:
        test_mean_entropy = np.mean(np.concatenate(test_entropies))
    else:
        test_mean_entropy = 0.0
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    ece = 0.0
    if not isinstance(model, (NNNet, FeaturesNet, KernelizedNet, LinearKernel)):
        test_logits = torch.cat(test_logits)
        test_labels = torch.cat(test_labels)
        ece = ece_meter(test_logits, test_labels).item()

    return misclassifications/totalSamples, ece, test_mean_entropy


if __name__ == "__main__":
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.EXP_NAME = args.exp_name
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    cfg.ACTIVE_LEARNING.MAX_ITER = args.max_iter
    cfg.ACTIVE_LEARNING.DELTA = args.delta
    cfg.ACTIVE_LEARNING.KERNEL = args.kernel
    cfg.ACTIVE_LEARNING.FEATURE = args.feature
    cfg.ACTIVE_LEARNING.HERDING_INIT = args.herding_init

    # uherding
    cfg.ACTIVE_LEARNING.UNC_TRANS_FN = args.unc_trans_fn
    cfg.ACTIVE_LEARNING.NORMALIZE = args.normalize
    cfg.ACTIVE_LEARNING.ADAPTIVE_DELTA = args.adaptive_delta
    cfg.ACTIVE_LEARNING.TEMP = 1.0

    cfg.RNG_SEED = args.seed
    cfg.MODEL.FINETUNE = False
    cfg.MODEL.LINEAR_FROM_FEATURES = args.linear_from_features
    cfg.MODEL.KERNELIZE = args.kernelize
    cfg.MODEL.NN = args.nn
    cfg.MODEL.ENT_WEIGHT = args.ent_weight
    cfg.MODEL.GAMMA = args.gamma
    cfg.MODEL.LAST_LAYER = args.last_layer
    if args.lr > 0.0:
        cfg.OPTIM.BASE_LR = args.lr

    args.loss = cfg.MODEL.LOSS_FUN
    main(cfg)
