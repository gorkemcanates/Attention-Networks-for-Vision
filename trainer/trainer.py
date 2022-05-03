__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Iterable
from termcolor import cprint


class MainTrainer:
    def __init__(self,
                 model,
                 params: object,
                 hyperparams: object,
                 metrics: list,
                 dataset: object,
                 optimizer: object,
                 criterion,
                 writer: object
                 ):
        self.model = model
        self.params = params
        self.hyperparams = hyperparams
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.writer = writer

        self.train_loader = DataLoader(dataset=dataset.train_dataset,
                                       batch_size=self.hyperparams.train_batch_size,
                                       shuffle=self.params.SHUFFLE)
        self.test_loader = DataLoader(dataset=dataset.test_dataset,
                                      batch_size=self.hyperparams.test_batch_size)

        self.DEVICE = torch.device('cuda:0'
                                   if self.params.DEVICE == 'cuda' else 'cpu')

        self.metric_keys = ['loss'] + \
                           [metric.__class__.__name__.lower()
                            for metric in self.metrics]

        self.train_history = {key: [] for key in self.metric_keys}
        self.test_history = {key: [] for key in self.metric_keys}
        self.loop = None
        self.predictions = None
        self.images = None
        self.targets = None
        self.epoch = 0
        self.best_metric = 100 if self.hyperparams.METRIC_CONDITION == 'min' else 0
        self.BEST = False
        self.BEGIN = True

        if self.params.PRETRAINED:
            checkpoint = self.load_model(to_device=True if self.params.DEVICE == 'cuda' else False)
            self.epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_checkpoint_metric']
        self.model = self.model.to(self.DEVICE)
        assert self.train_loader and self.test_loader is not None

    def fit(self):
        if not isinstance(self.train_loader, Iterable):
            raise Exception("Train loader must be an iterable")
        try:
            for epoch in range(self.epoch,
                               self.hyperparams.NUM_EPOCHS + self.epoch):
                self.epoch = epoch
                self.loop = tqdm(enumerate(self.train_loader),
                                 total=len(self.train_loader),
                                 leave=False)
                self.adjust_lr(epoch=epoch,
                               case=self.hyperparams.lr_case)
                epoch_training_results = self.fit_one_epoch()
                self.store(epoch_training_results,
                           'train')
                self.print_epoch('train')

                if self.params.VALIDATION:
                    epoch_val_results = self.validate()
                    self.store(epoch_val_results,
                               'test')
                    self.print_epoch('test')
                    self.update_model()
                    self.update_best_metric(condition=self.hyperparams.METRIC_CONDITION)

                    if self.params.TO_TENSORBOARD:
                        self.writer.write_results(self.metric_keys,
                                                  epoch_training_results,
                                                  epoch_val_results,
                                                  self.epoch,
                                                  )

            if self.params.TO_TENSORBOARD:
                self.writer.flush()
                self.writer.close()
        except KeyboardInterrupt:
            print('Training terminated.')

    def fit_one_epoch(self):
        self.model.train()

        total_batch_results = torch.zeros(len(self.metric_keys), device=self.DEVICE)
        count = 0
        for _, (data, target) in self.loop:
            data, target = data.to(self.DEVICE), target.to(self.DEVICE)
            batch_loss, preds = self.fit_one_batch(data, target)
            batch_metrics = self.get_metrics(preds, target)
            batch_results = torch.cat((torch.tensor(batch_loss,
                                                    device=self.DEVICE).reshape(1, ),
                                       batch_metrics), dim=0)

            if batch_results[1] < batch_results[2]:
                raise AssertionError('Top-1 error cannot be smaller than Top-5 error.')

            self.print_batch(batch_results)
            total_batch_results += batch_results
            count += 1
        epoch_results = total_batch_results / count
        return epoch_results

    def fit_one_batch(self,
                      data,
                      target,
                      grad=True):
        self.optimizer.zero_grad()
        preds = self.forward(data)
        if self.params.DEVICE == 'cuda':
            target = target.type(torch.cuda.FloatTensor)
        else:
            target = target.type(torch.FloatTensor)
        loss = self.loss_function(preds, target)
        if grad:
            self.backward(loss)
        return loss, preds

    def validate(self):
        self.model.eval()
        if not isinstance(self.test_loader, Iterable):
            raise Exception("Test loader must be an iterable")
        with torch.no_grad():
            total_val_batch_results = torch.zeros(len(self.metric_keys),
                                                  device=self.DEVICE)
            count = 0
            predictions_list = []
            image_list = []
            target_list = []
            for _, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                preds = self.forward(data)
                if self.params.DEVICE == 'cuda':
                    target = target.type(torch.cuda.FloatTensor)
                else:
                    target = target.type(torch.FloatTensor)
                val_batch_loss = self.loss_function(preds, target)
                val_batch_metrics = self.get_metrics(preds, target)
                val_batch_results = torch.cat((torch.tensor(val_batch_loss,
                                                            device=self.DEVICE).reshape(1, ),
                                               val_batch_metrics),
                                              dim=0)
                total_val_batch_results += val_batch_results
                count += 1
                preds = self.evaluate(preds)
                predictions_list.append(preds.detach().cpu())
                if self.BEGIN:
                    image_list.append(data.detach().cpu())
                    target_list.append(target.detach().cpu())

                epoch_val_results = total_val_batch_results / count

            self.predictions = torch.cat(predictions_list, dim=0)
            if self.BEGIN:
                self.images = torch.cat(image_list, dim=0)
                self.targets = torch.cat(target_list, dim=0)
                self.BEGIN = False

            return epoch_val_results

    def forward(self,
                data):
        return self.model(data)

    def backward(self,
                 loss):
        loss.backward()
        self.optimizer.step()

    def loss_function(self,
                      preds,
                      target):
        return self.criterion(preds, target)

    def adjust_lr(self,
                  epoch,
                  case='ImageNet'):

        if case == 'ImageNet':
            lr = self.hyperparams.LEARNING_RATE * (0.1 ** (epoch // 30))
        elif case == 'CIFAR':
            if epoch >=100:
                lr = self.hyperparams.LEARNING_RATE * (0.1 ** (epoch // 50 - 1))
            else:
                lr = self.hyperparams.LEARNING_RATE
        else:
            lr = self.hyperparams.LEARNING_RATE
        for param in self.optimizer.param_groups:
            param['lr'] = lr

    def evaluate(self,
                 preds,
                 ):
        return torch.argmax(F.softmax(preds), dim=1)

    def get_metrics(self,
                    pred,
                    target):
        local_container = [metric(pred, target) for metric in self.metrics]
        batch_metrics = torch.tensor(local_container, device=self.DEVICE)
        return batch_metrics

    def store(self,
              epoch_results,
              flag):
        i = 0
        for key in self.metric_keys:
            if flag == 'train':
                self.train_history[key].append(self.round_metrics(epoch_results[i].detach().cpu(), 4))
            elif flag == 'test':
                self.test_history[key].append(self.round_metrics(epoch_results[i].detach().cpu(), 4))
            else:
                raise Exception(f'flag must be either train or test.')
            i += 1

    def round_metrics(self,
                      result,
                      dec):
        return torch.round(result * 10 ** dec) / (10 ** dec)

    def print_batch(self,
                    results):
        print_dict = {key: self.round_metrics(results[i].detach().cpu(), 4).numpy()
                      for key, i in zip(self.metric_keys, range(len(results)))}
        self.loop.set_description(f'Epoch-> {self.epoch + 1}')
        self.loop.set_postfix(ordered_dict=print_dict)

    def print_epoch(self,
                    stage):
        if stage == 'train':
            print_dict = {key: round(float(self.train_history[key][-1].detach().cpu()), 4)
                          for key in self.metric_keys}
            print(f'\n''Training -- Epoch:' + repr(self.epoch + 1) + ' --> ' + str(print_dict))

        elif stage == 'test':
            print_dict = {key: round(float(self.test_history[key][-1].detach().cpu()), 4)
                          for key in self.metric_keys}
            cprint('Validation -- Epoch:' + repr(self.epoch + 1) + ' --> ' + str(print_dict), 'blue')
            if self.test_history[self.params.METRIC_CONDITION][-1] < self.best_metric:
                cprint(f'Best: {round(float(self.test_history[self.params.METRIC_CONDITION][-1]), 4)} --> '
                       f'Last: {round(self.best_metric, 4)}',
                       'cyan',
                       attrs=['bold'])
        else:
            raise Exception(f'stage must be either Training or Test.')

    def update_best_metric(self,
                           condition='max'):
        if condition == 'max':
            self.best_metric = np.max((self.best_metric,
                                       self.test_history[self.params.METRIC_CONDITION][-1]))
        elif condition == 'min':
            self.best_metric = np.min((self.best_metric,
                                       self.test_history[self.params.METRIC_CONDITION][-1]))
        else:
            raise Exception('Best metric update condition must be either min or max.')


    def save_results(self):
        if not os.path.exists(self.params.result_SAVEPATH):
            os.makedirs(self.params.result_SAVEPATH)
        np.save(os.path.join(
            self.params.result_SAVEPATH, 'train_results.npy'),
            self.train_history)
        np.save(os.path.join(
            self.params.result_SAVEPATH, 'test_results.npy'),
            self.test_history)

    def update_model(self):
        if not os.path.exists(self.params.model_SAVEPATH):
            os.makedirs(self.params.model_SAVEPATH)
        self.BEST = self.test_history[self.params.METRIC_CONDITION][-1] < self.best_metric
        if self.BEST:
            torch.save(
                {
                    'epoch': self.epoch,
                    'total_params': sum(p.numel() for p in self.model.parameters()),
                    'best_checkpoint_metric': self.best_metric,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },
                os.path.join(self.params.model_SAVEPATH,
                             'model.pth')
            )
            self.save_results()

    def load_model(self,
                   to_device=torch.cuda.is_available(),
                   final_model=False):
        checkpoint = torch.load(os.path.join(
            self.params.model_LOADPATH, 'model.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if to_device:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        epoch = checkpoint['epoch']
        print(f'Pretrained model is loaded from epoch: {epoch}')
        if to_device:
            self.model = self.model.to(self.DEVICE)

        if final_model:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        return checkpoint