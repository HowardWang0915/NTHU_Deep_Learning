import argparse
import torch
import utils
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from tqdm import tqdm
from parse_config import ConfigParser


def main(config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        gen_data = []
        gen_label = []
        classes = []
        for _, (data, target) in enumerate(tqdm(data_loader)):
            label_name = data_loader.label_name[target[0].item()]
            # retreive the mother image and labels from data_loader
            mother_img = data.squeeze(0)
            data = data.to(device, dtype=torch.float)
            latent, _ = model(data)
            gen_img_list = []   # save the generated images
            for _ in range(5):
                # add some gaussian noise into the latent code
                noised = latent + torch.normal(mean=0., std=0.1, size=latent.size()).to(device)
                gen_img = model(noised, decode_only=True)
                gen_img = gen_img.squeeze(0)
                gen_img_list.append(gen_img.data)
                gen_img = gen_img.permute(1, 2, 0)
                gen_data.append(gen_img.data.cpu().numpy())
                gen_label.append([target.data.cpu().numpy()])
            if label_name not in classes:
                utils.imshow(mother_img, gen_img_list, label_name)
                classes.append(label_name)
                gen_img_list.clear()
        gen_data = np.array(gen_data)
        logger.info(str(len(gen_data)) + "images successfully generated")
        logger.info("Saving data to file")
        np.save(str(config.log_dir) + '/gen_data', gen_data)
        gen_label = np.array(gen_label)
        np.save(str(config.log_dir) + '/gen_label', gen_label)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
