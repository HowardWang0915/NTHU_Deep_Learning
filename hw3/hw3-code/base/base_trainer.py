import numpy as np
from abc import abstractmethod

class BaseTrainer:
    """
        Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        
        trainer_cfg = config['trainer']
        self.epochs = trainer_cfg['epochs']
        self.save_period = trainer_cfg['save_period']
        self.monitor = trainer_cfg.get('monitor', 'off')

        # Configuration to monitor model performance and save best
        # Don't save the best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = np.inf if self.mnt_mode == 'min' else - np.inf
            self.early_stop = trainer_cfg.get('early_stop', np.inf)
            if self.early_stop <= 0:
                self.early_stop = np.inf
        
        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        

    @abstractmethod
    def _train_epoch(self, epoch):
        """
            Training logic for an epoch
        """
        raise NotImplementedError

    def train(self):
        """
            Full training logic
        """
        not_improved_count = 0
        losses = {'Training loss': [], 'Validation loss': []}
        accuracies = {'Training acc': [], 'Validation acc': []}
        steps = []
        # Loop through each epoch
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Train logic for custom data loader
            result = self._train_epoch(epoch)

            # record results
            steps.append(epoch)
            losses['Training loss'].append(result['loss'])
            losses['Validation loss'].append(result['val_loss'])
            #  import pdb; pdb.set_trace()
            accuracies['Training acc'].append(result['accuracy'])
            accuracies['Validation acc'].append(result['val_accuracy'])

            # save logged infos
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to console
            for key, value in log.items():
                self.logger.info('   {:15s}: {}'.format(str(key), value))

            # model evaluation
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)

                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                
                if improved:
                    self.mnt_mode = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))

                    break
                
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        return losses, accuracies, steps
                

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.npz'.format(epoch))
        np.savez(filename, **self.model.weights)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.npz')
            np.savez(best_path, **self.model.weights)
            self.logger.info("Saving current best: model_best.npz ...")
