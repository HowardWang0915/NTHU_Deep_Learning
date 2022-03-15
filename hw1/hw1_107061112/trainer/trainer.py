import numpy as np 
from base import BaseTrainer
from utils import MetricTracker

"""
    This file defines all custom trainers. To add a trainer, write a class
    and inherit the BaseTrainer class
"""

class MNIST_Trainer(BaseTrainer):
    """
        A trainer for MNIST dataset
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
            lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=None)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=None)

    def _train_epoch(self, epoch):
        """
            Training logic for an epoch
        """
        # SGD, using mini-batch
        loss = self.criterion()
        # Iterate through mini-batches
        for batch_idx, (data, target) in enumerate(self.data_loader):
            outputs = self.model.forward(data)
            cost = loss.forward(outputs, target)
            # Backprop value of the loss
            L_bp = loss.backward(outputs, target)
            # Pass the loss backward to perform backprop
            gradients = self.model.backward(L_bp, self.data_loader.batch_size)
            self.model.optimize(gradients)

            # update loss value
            self.train_metrics.update('loss', float(cost))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(outputs, target))

            # print results to terminal
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    float(cost)))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        # Do validation
        if self.do_validation:
            val_log = self._validation()
            log.update(**{'val_'+k : v for k, v in val_log.items()})
        return log

    def _validation(self):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        """
        self.valid_metrics.reset()
        loss = self.criterion()
        data = self.data_loader.X_val
        target = self.data_loader.y_val

        outputs = self.model.forward(data)
        cost = loss.forward(outputs, target)

        self.valid_metrics.update('loss', float(cost))
        for met in self.metric_ftns:
            self.valid_metrics.update(met.__name__, met(outputs, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        """ 
            Handling the log format for logging
            :batch_idx: the index to be logged
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
