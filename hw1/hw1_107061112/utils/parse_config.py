import os
import logging
from datetime import datetime
from logger import setup_logging
from pathlib import Path
from utils import read_json

class ConfigParser:
    def __init__(self, config, resume=None, run_id=None):
        """
            Class to parse configuration json file. Handles hyperparameter for 
            training, initializations of modules and loggin module

            Params:
                Config: A json dict containing configs, hyperparameter for 
                    training, contents of 'config.json'.
        """
        # load config file
        self._config = config
        self.resume = resume

        # model and log file dir
        save_dir = Path(self.config['trainer']['save_dir'])

        # Experiment name
        exper_name = self.config['name']
        
        # use timestamp as run-id for easy logging
        if run_id is None:
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')

        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make dir for saving logs using run id
        exist_ok = run_id == ''
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args):
        """
            Initialize this class from some cli arguments. Used in train, test
        """
        #  for opt in options:
            #  args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = Path(args.config)
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            resume = None
            assert args.config is not None, msg_no_cfg
            cfg_fname = Path(args.config)

        config = read_json(cfg_fname)
        
        if args.config:
            print('Json finished loading!')
        
        return cls(config, resume)
        
    def init_obj(self, name, module, *args, **kwargs):
        """
            Finds a function handle with the name given as 'type' in config,
            and returns the instance initialized with corresponding arguments given.

            For example:
            object = config.init_obj('name', module, a, b=1)
            this will initialize module.name(a, b=1)
        """

        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
       
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        """
            Access items like ordinary dict.
        """
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(\
                verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # Several properties are read-only
    @property
    def config(self):
        return self._config

    @property 
    def save_dir(self):
        return self._save_dir
        
    @property 
    def log_dir(self):
        return self._log_dir
