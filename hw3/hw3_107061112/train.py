import argparse
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.optimizer as module_optimizer
import model.model as module_arch
from trainer import Fruit_Trainer
from utils.parse_config import ConfigParser
from utils import plot_learning_curve

def main(config):
    """ 
        Main function to train a model
        :config: Configuration parsed from config.json
    """

    # setup train logger
    logger = config.get_logger('train')
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # Build model instances
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    optimizer = config.init_obj('optimizer', module_optimizer)
    lr_scheduler = None


    trainer = Fruit_Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        data_loader=data_loader,
                        lr_scheduler=lr_scheduler)
    losses, acc, step = trainer.train()

    plot_learning_curve(string="Loss", step=step, data=losses, x=1)
    plot_learning_curve(string="Accuracy", step=step, data=acc, x=2)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='HW1 MNIST Shallow Network Train')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    # load the config file from command line
    config = ConfigParser.from_args(args)

    main(config)

    
