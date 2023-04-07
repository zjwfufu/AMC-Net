import os
import random

import numpy as np
import torch

from models.model import AMC_Net


def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_exp_settings(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 20)
    log_dict = cfg.__dict__.copy()
    for k, v in log_dict.items():
        logger.info(f'{k} : {v}')
    logger.info('=' * 20)


def create_AMC_Net(cfg):
    """
    build AWN model
    """
    model = AMC_Net(
        num_classes=cfg.num_classes,
        sig_len=cfg.sig_len,
        extend_channel=cfg.extend_channel,
        latent_dim=cfg.latent_dim,
        num_heads=cfg.num_heads,
        conv_chan_list=cfg.conv_chan_list
    ).to(cfg.device)

    return model
