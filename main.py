from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import json
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  #pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)

from timm.scheduler.cosine_lr import CosineLRScheduler

def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.
    Adds smoothing to the target distribution to prevent overconfidence in predictions.
    """
    def __init__(self, smoothing=0.1):
        """
        Args:
            smoothing (float): Smoothing factor in the range [0.0, 1.0].
                              0.0 means no smoothing (standard cross-entropy).
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing value must be in the range [0, 1)"
        self.smoothing = smoothing

    def forward(self, logits, target):
        """
        Forward pass for the loss computation.

        Args:
            logits (Tensor): Model output logits of shape [B, num_classes].
            target (Tensor): Target labels of shape [B]. Must contain class indices.

        Returns:
            Tensor: Scalar loss value.
        """
        num_classes = logits.size(-1)

        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed labels
        with torch.no_grad():
            # Initialize label smoothing distribution
            smoothed_labels = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smoothed_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Compute the negative log likelihood loss
        loss = -torch.sum(smoothed_labels * log_probs, dim=-1)
        return loss.mean()


def get_parser():
    parser = argparse.ArgumentParser(description='SkateFormer: Skeletal-Temporal Transformer for Human Action Recognition')

    # General arguments
    parser.add_argument('--work-dir', default='./work_dir', help='The work folder for storing results')
    parser.add_argument('--model-saved-name', default='', help='Filename for saving the trained model')
    parser.add_argument('--config', default='./config', help='Path to the configuration file')
    parser.add_argument('--phase', default='train', help='Phase: train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='Save the classification score')

    # Logging and debugging
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--log-interval', type=int, default=100, help='Interval for printing logs (iterations)')
    parser.add_argument('--save-interval', type=int, default=1, help='Interval for saving models (epochs)')
    parser.add_argument('--save-epoch', type=int, default=30, help='Starting epoch to save models')
    parser.add_argument('--eval-interval', type=int, default=5, help='Evaluation interval (epochs)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='Print logs during training')
    parser.add_argument('--show-topk', type=int, nargs='+', default=[1, 5], help='Top-K accuracies to display')

    # Data loader
    parser.add_argument('--feeder', default='feeder.feeder', help='Data loader module')
    parser.add_argument('--num-worker', type=int, default=4, help='Number of workers for the data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='Arguments for the training data loader')
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(), help='Arguments for the test data loader')

    # Model-specific arguments
    parser.add_argument('--model', default=None, help='Name of the model to use')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='Arguments for configuring the model')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 6, 2], help='Depths of each stage in the model')
    parser.add_argument('--channels', type=int, nargs='+', default=[96, 192, 384, 768], help='Channels for each stage in the model')
    parser.add_argument('--embed-dim', type=int, default=96, help='Embedding dimension for the input layer')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff-dim', type=int, default=512, help='Dimension of the feedforward layers in transformer blocks')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for attention and feedforward layers')
    parser.add_argument('--drop-path', type=float, default=0.1, help='Drop path rate for stochastic depth')
    parser.add_argument('--num-classes', type=int, default=60, help='Number of classes for classification')
    parser.add_argument('--global-pool', type=str, choices=['avg', 'max'], default='avg', help='Type of global pooling (avg or max)')
    parser.add_argument('--weights', default=None, help='Path to pretrained weights for the model')
    parser.add_argument('--ignore-weights', type=str, nargs='+', default=[], help='List of weight names to ignore during initialization')

    # Optimization arguments
    parser.add_argument('--nesterov', type=bool, default=False, help='Use Nesterov momentum (for SGD optimizer)') 
    parser.add_argument('--optimizer', default='AdamW', help='Type of optimizer to use')
    parser.add_argument('--warm-up-epoch', type=int, default=5, help='Number of warm-up epochs')
    parser.add_argument('--warmup-prefix', type=bool, default=False, help='Apply warm-up to initial phase only')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate for learning rate scheduler')
    parser.add_argument('--base-lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay for regularization')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--grad-clip', type=str2bool, default=False, help='Enable gradient clipping')
    parser.add_argument('--grad-max', type=float, default=1.0, help='Maximum gradient norm for clipping')

    # Learning rate scheduler
    parser.add_argument('--lr-scheduler', default='cosine', help='Learning rate scheduler type')
    parser.add_argument('--t-max', type=int, default=80, help='Total iterations for cosine annealing scheduler')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate for the scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, help='Learning rate during warmup')

    # Loss function
    parser.add_argument('--loss-type', type=str, default='CE', help='Loss type: CE (CrossEntropy) or LabelSmoothing')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Smoothing factor for Label Smoothing')

    # Device configuration
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='Indexes of GPUs to use for training/testing')

    # Training schedule
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch for training')
    parser.add_argument('--num-epoch', type=int, default=80, help='Total number of epochs for training')
    parser.add_argument('--eval-epoch', type=int, default=5, help='Evaluate every n epochs')

    return parser



class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')

        self.global_step = 0
        self.load_model()
        self.load_data()

        if self.arg.phase == 'train':
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader['train']))

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.model = self.model.to(device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    
    
    def load_data(self):
        Feeder = import_class(self.arg.feeder)  # Dynamically import the specified feeder class
        self.data_loader = dict()

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),  # Pass feeder arguments
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed
            )
        
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),  # Pass feeder arguments
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed
        )

    def load_model(self):
        """
        Load the SkateFormer model and optionally load pretrained weights.
        """
        # Set the output device
        output_device = self.arg.device[0] if self.arg.device else 'cpu'
        self.output_device = output_device

        # Import and initialize the SkateFormer model
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(f"Using model: {Model.__name__}")

        # Initialize the model with arguments
        self.model = Model(
            depths=self.arg.model_args.get('depths', [2, 2, 6, 2]),
            channels=self.arg.model_args.get('channels', [96, 192, 384, 768]),
            embed_dim=self.arg.model_args.get('embed_dim', 96),
            num_heads=self.arg.model_args.get('num_heads', 8),
            ff_dim=self.arg.model_args.get('ff_dim', 512),
            dropout=self.arg.model_args.get('dropout', 0.1),
            drop_path=self.arg.model_args.get('drop_path', 0.1),
            num_classes=self.arg.num_classes,
            global_pool=self.arg.model_args.get('global_pool', 'avg'),
            index_t=self.arg.model_args.get('index_t', False)
        ).to(torch.device(self.output_device))

        # Initialize the loss function
        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().to(torch.device(self.output_device))
        elif self.arg.loss_type == 'LabelSmoothing':
            smoothing = self.arg.model_args.get('smoothing', 0.1)
            self.loss = LabelSmoothingCrossEntropy(smoothing=smoothing).to(torch.device(self.output_device))
        elif self.arg.loss_type == 'LSCE':  # Add LSCE support
            smoothing = self.arg.model_args.get('smoothing', 0.1)
            self.loss = LabelSmoothingCrossEntropy(smoothing=smoothing).to(torch.device(self.output_device))
        else:
            raise ValueError(f"Unsupported loss type: {self.arg.loss_type}")

        # Load weights
        if self.arg.weights:
            self.print_log(f"Loading weights from {self.arg.weights}...")
            try:
                weights_path = self.arg.weights
                weights = torch.load(weights_path, map_location=torch.device(self.output_device))

                # Get the model's state dictionary
                model_dict = self.model.state_dict()

                # Filter the checkpoint weights
                pretrained_dict = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].shape == v.shape}
                ignored_keys = [k for k in weights.keys() if k not in pretrained_dict]
                missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]

                # Log ignored and missing keys
                self.print_log(f"Ignored keys (from checkpoint): {ignored_keys}")
                self.print_log(f"Missing keys (in model): {missing_keys}")

                # Load the filtered weights
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)

                # Reinitialize incompatible layers if required
                for key in missing_keys:
                    if 'stem' in key:
                        self.print_log(f"Reinitializing: {key}")
                        self.model.stem = nn.Conv2d(3, 96, kernel_size=1, stride=1, padding=0, bias=False)
                    elif 'head' in key:
                        self.print_log(f"Reinitializing: {key}")
                        self.model.head = nn.Linear(768, 60)

                self.print_log("Weights loaded successfully.")

            except Exception as e:
                self.print_log(f"Error loading weights: {e}")
                raise e


    def load_optimizer(self):
        """
        Load the optimizer for the spatio-temporal transformer-based model.
        Supports SGD, Adam, and AdamW.
        """
        if self.arg.optimizer == 'SGD':
            # Stochastic Gradient Descent (SGD)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'Adam':
            # Adam Optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        elif self.arg.optimizer == 'AdamW':
            # AdamW Optimizer (Recommended for transformer models)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.arg.optimizer}")

        # Log warm-up configuration if enabled
        if self.arg.warm_up_epoch > 0:
            self.print_log(f"Using learning rate warm-up for {self.arg.warm_up_epoch} epochs.")
        else:
            self.print_log("No warm-up configured.")

    def load_scheduler(self, n_iter_per_epoch):
        """
        Load the learning rate scheduler for the spatio-temporal transformer-based model.
        Supports cosine annealing with warm-up.
        """
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)  # Total number of iterations
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)  # Warm-up iterations

        if self.arg.lr_scheduler == 'cosine':
            # Cosine Annealing with Warm-Up
            self.lr_scheduler = CosineLRScheduler(
                optimizer=self.optimizer,
                t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
                lr_min=self.arg.lr_min,           # Minimum learning rate after decay
                warmup_lr_init=self.arg.warmup_lr,  # Starting learning rate during warm-up
                warmup_t=warmup_steps,            # Warm-up duration (iterations)
                cycle_limit=1,                    # Single cosine cycle
                t_in_epochs=False,                # Scheduler operates in iterations, not epochs
                warmup_prefix=self.arg.warmup_prefix
            )
            self.print_log(f"Using cosine LR scheduler with {warmup_steps} warm-up steps.")
        else:
            raise ValueError(f"Unsupported LR scheduler type: {self.arg.lr_scheduler}")



    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)
    

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    

    def train(self, epoch, save_model=True):
        self.model.train()  # Set the model to training mode
        self.print_log(f"Training epoch: {epoch + 1}")
        loader = self.data_loader['train']

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)  # Log the current epoch
        self.record_time()

        # Timer to track time usage
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, dynamic_ncols=True)  # Progress bar for the training loop

        for batch_idx, (data, index_t, label, index) in enumerate(process):
            # Update learning rate for the current step
            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1

            # Load data to GPU/CPU
            with torch.no_grad():
                data = data.float().to(self.output_device)  # Shape: [B, C, T, V, M]
                index_t = index_t.float().to(self.output_device)  # Temporal positional embedding indices
                label = label.long().to(self.output_device)  # Ground truth labels
            timer['dataloader'] += self.split_time()

            # Forward pass
            output = self.model(data, index_t)  # Model output
            loss = self.loss(output, label)  # Compute loss
            timer['model'] += self.split_time()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.arg.grad_clip:
                # Clip gradients to avoid exploding gradients in transformers
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
            self.optimizer.step()

            # Track training loss and accuracy
            loss_value.append(loss.item())
            value, predict_label = torch.max(output.data, 1)  # Get predicted labels
            acc = torch.mean((predict_label == label).float())  # Compute accuracy
            acc_value.append(acc.item())

            # Log metrics to TensorBoard
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.item(), self.global_step)

            # Record learning rate
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)

            # Update timer
            timer['statistics'] += self.split_time()

            # Update progress bar
            process.set_description(f"Loss: {loss.item():.4f}, Acc: {acc.item() * 100:.2f}%, LR: {self.lr:.6f}")

        # Log epoch statistics
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%' for k, v in timer.items()
        }
        mean_loss = np.mean(loss_value)
        mean_acc = np.mean(acc_value) * 100
        self.print_log(
            f"\tMean training loss: {mean_loss:.4f}. Mean training acc: {mean_acc:.2f}%."
        )
        self.print_log(f"\tLearning Rate: {self.lr:.4f}")
        self.print_log(
            f"\tTime consumption: [Data]{proportion['dataloader']}, [Network]{proportion['model']}, [Stats]{proportion['statistics']}"
        )

        # Save model if required
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict(
                [[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()]
            )
            model_path = f"{self.arg.model_saved_name}-{epoch + 1}-{self.global_step}.pt"
            torch.save(weights, model_path)
            self.print_log(f"Model saved to {model_path}")

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        """
        Evaluation loop for the spatio-temporal transformer-based model.

        Args:
            epoch (int): Current epoch.
            save_score (bool): Whether to save scores for each sample.
            loader_name (list): List of data loaders to evaluate (e.g., ['test']).
            wrong_file (str): Path to save misclassified samples.
            result_file (str): Path to save predicted vs true labels.
        """
        # Initialize files for saving wrong predictions or results
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        self.model.eval()  # Set model to evaluation mode
        self.print_log(f"Eval epoch: {epoch + 1}")

        for ln in loader_name:
            # Initialize metrics and storage
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], dynamic_ncols=True)

            for batch_idx, (data, index_t, label, index) in enumerate(process):
                label_list.append(label)  # Store true labels
                with torch.no_grad():
                    # Prepare data
                    data = data.float().to(self.output_device)  # Shape: [B, C, T, V, M]
                    index_t = index_t.float().to(self.output_device)  # Temporal indices
                    label = label.long().to(self.output_device)  # True labels

                    # Forward pass
                    output = self.model(data, index_t)  # Model output
                    loss = self.loss(output, label)  # Compute loss

                    # Store scores and loss
                    score_frag.append(output.data.cpu().numpy())  # Scores for each sample
                    loss_value.append(loss.item())

                    # Get predictions
                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.cpu().numpy())

                    step += 1

                    # Save wrong predictions or results if specified
                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.cpu().numpy())
                        for i, pred in enumerate(predict):
                            if result_file is not None:
                                f_r.write(f"{pred},{true[i]}\n")
                            if pred != true[i] and wrong_file is not None:
                                f_w.write(f"{index[i]},{pred},{true[i]}\n")

            # Aggregate results
            score = np.concatenate(score_frag)  # Combine scores across batches
            loss = np.mean(loss_value)  # Mean loss
            label_list = np.concatenate(label_list)  # True labels
            pred_list = np.concatenate(pred_list)  # Predicted labels

            # Compute accuracy
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)  # Top-1 accuracy
            if accuracy > self.best_acc:  # Track the best accuracy
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            # Log accuracy
            self.print_log(f"Accuracy: {accuracy:.2f}%, Model: {self.arg.model_saved_name}")

            # Log validation metrics if in training phase
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            # Save scores if required
            if save_score:
                score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
                score_file = f"{self.arg.work_dir}/epoch{epoch + 1}_{ln}_score.pkl"
                with open(score_file, 'wb') as f:
                    pickle.dump(score_dict, f)

            # Log mean loss and Top-K accuracy
            self.print_log(f"\tMean {ln} loss: {loss:.4f}")
            for k in self.arg.show_topk:
                top_k_acc = self.data_loader[ln].dataset.top_k(score, k)
                self.print_log(f"\tTop-{k} accuracy: {top_k_acc * 100:.2f}%")

            # Class-wise accuracy and confusion matrix
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum

            # Save class-wise accuracy and confusion matrix
            csv_file = f"{self.arg.work_dir}/epoch{epoch + 1}_{ln}_each_class_acc.csv"
            with open(csv_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Class Accuracy'] + list(each_acc))
                writer.writerows(confusion)

        # Close files if used
        if wrong_file is not None:
            f_w.close()
        if result_file is not None:
            f_r.close()

    
    def start(self):
        """
        Main entry point for training or testing the spatio-temporal transformer model.
        Handles training, evaluation, and logging.
        """
        if self.arg.phase == 'train':
            # Log parameters and initialize global step
            self.print_log(f"Parameters:\n{str(vars(self.arg))}\n")
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])

            # Function to count trainable parameters
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f"# Parameters: {count_parameters(self.model)}")

            # Training loop
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                # Save models only in the later epochs (e.g., last 10%)
                save_model = (epoch + 1) >= self.arg.num_epoch * 0.9

                # Train and evaluate
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=save_model, loader_name=['test'])

            # Load the best model based on accuracy
            weights_path = glob.glob(os.path.join(self.arg.work_dir, f"runs-{self.best_acc_epoch}*"))[0]
            self.print_log(f"Loading best model weights from: {weights_path}")
            weights = torch.load(weights_path)

            # Map weights for multi-GPU support, if necessary
            if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.to(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            # Evaluate the best model
            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False  # Temporarily disable logging for evaluation
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True  # Re-enable logging

            # Log final results
            num_params = count_parameters(self.model)
            self.print_log(f"Best accuracy: {self.best_acc:.2f}%")
            self.print_log(f"Epoch with best accuracy: {self.best_acc_epoch}")
            self.print_log(f"Model directory: {self.arg.work_dir}")
            self.print_log(f"Total trainable parameters: {num_params}")
            self.print_log(f"Weight decay: {self.arg.weight_decay}")
            self.print_log(f"Base learning rate: {self.arg.base_lr}")
            self.print_log(f"Training batch size: {self.arg.batch_size}")
            self.print_log(f"Test batch size: {self.arg.test_batch_size}")
            self.print_log(f"Random seed: {self.arg.seed}")

        elif self.arg.phase == 'test':
            # Ensure weights are provided for testing
            if self.arg.weights is None:
                raise ValueError("Please specify --weights for testing.")

            # Prepare output files for wrong predictions and results
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            # Log model and weights information
            self.print_log(f"Model:   {self.arg.model}")
            self.print_log(f"Weights: {self.arg.weights}")

            # Load the specified weights
            weights = torch.load(self.arg.weights)
            if isinstance(self.arg.device, list) and len(self.arg.device) > 1:
                weights = OrderedDict([['module.' + k, v.to(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            # Evaluate the model
            self.arg.print_log = False  # Temporarily disable logging for evaluation
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True  # Re-enable logging

            self.print_log("Testing completed.\n")

if __name__ == '__main__':
    # Parse arguments
    parser = get_parser()
    p = parser.parse_args()

    # Load configuration file
    if p.config is not None:
        print(f"Loading configuration from {p.config}...")
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)  # Use safe_load for better security

        # Validate configuration keys
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f"WARNING: Unrecognized argument '{k}' in configuration.")
                assert k in key, f"Unrecognized argument: {k}"
        parser.set_defaults(**default_arg)
    # Merge command-line arguments with configuration
    arg = parser.parse_args()

    # Log merged arguments
    print("Final arguments:")
    print(json.dumps(vars(arg), indent=4)) 
    # Initialize seed for reproducibility
    init_seed(arg.seed)

    # Run processor
    try:
        processor = Processor(arg)
        processor.start()
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

