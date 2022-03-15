import argparse
import data_loader.data_loaders as module_data
import numpy as np
import model.loss as module_loss
import model.model as module_arch
import model.metric as module_metric
from utils.parse_config import ConfigParser

def main(config):
    """ 
        Main function to test our best model
        :config: Configuration parsed from config.json
    """
    logger = config.get_logger('test')

    # Set up data loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size = None,
        shuffle = False,
        val_split = 0.0,
        training = False,
        n_workers = 2
    )
    
    # Build model instances
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # Load saved parameters
    model.resume(config.resume)
    logger.info("Loading checkpoint: {} ...".format(config.resume))
    
    # start computing the loss and acc on our best model with test data
    total_loss = 0.0
    total_metrics = np.zeros(len(metric_fns))
    data = data_loader.X_test
    target = data_loader.y_test

    # Do foward pass and calculat loss
    outputs = model.forward(data)
    loss = loss_fn()
    cost = loss.forward(outputs, target)
    total_loss += cost
    # Calculate results using metrics in config.json
    for i, metric in enumerate(metric_fns):
        total_metrics[i] += metric(outputs, target)

    log = {'loss': total_loss}
    log.update({
        met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)
    })
    # Send it to logger
    logger.info(log)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='HW1 MNIST Shallow Network Test')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    # load the config file from command line
    config = ConfigParser.from_args(args)
    main(config)
