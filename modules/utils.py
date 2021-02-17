from modules.angle_clip import AngleClipper
from modules.angle_prior import AnglePriorsLoss
from modules.intersect import IntersectLoss
from modules.body_prior import BodyPrior
from modules.angle_sum import AngleSumLoss
from modules.change_loss import ChangeLoss
from model import VPoserModel
import smplx
import re


def is_loss_enabled(config, name):
    return config['pose'][name]['enabled']


def toggle_loss_enabled(config, name, value):
    config['pose'][name]['enabled'] = value


def get_loss_conf(config, name):
    return config['pose'][name]


def camel_to_snake(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def get_layer_config(config, name):
    params = {}

    layer_conf = config['pose'][name]

    for k, v in layer_conf.items():
        if k == 'enabled':
            continue
        params[camel_to_snake(k)] = v

    return params


def get_loss_layers(config, model: smplx.SMPL, device, dtype):
    """ Utility method to create loss layers based on a config file

    Args:
        config ([type]): [description]
        device ([type]): [description]
        dtype ([type]): [description]
    """
    extra_loss_layers = []

    if config['pose']['bodyPrior']['enabled']:

        vmodel = VPoserModel.from_conf(config)
        extra_loss_layers.append(BodyPrior(
            device=device,
            dtype=dtype,
            vmodel=vmodel,
            weight=config['pose']['bodyPrior']['weight']))

    if config['pose']['anglePrior']['enabled']:
        extra_loss_layers.append(AnglePriorsLoss(
            device=device,
            global_weight=config['pose']['anglePrior']['weight'],
            dtype=dtype))

    if config['pose']['angleSumLoss']['enabled']:
        extra_loss_layers.append(AngleSumLoss(
            device=device,
            dtype=dtype,
            weight=config['pose']['angleSumLoss']['weight']))

    if config['pose']['angleLimitLoss']['enabled']:
        extra_loss_layers.append(AngleClipper(
            device=device,
            dtype=dtype,
            weight=config['pose']['angleLimitLoss']['weight']))

    if config['pose']['intersectLoss']['enabled']:
        extra_loss_layers.append(IntersectLoss(
            model=model,
            device=device,
            dtype=dtype,
            weight=config['pose']['intersectLoss']['weight'],
            sigma=config['pose']['intersectLoss']['sigma'],
            max_collisions=config['pose']['intersectLoss']['maxCollisions']
        ))

    changeLoss = get_loss_conf(config, "changeLoss")
    if changeLoss['enabled']:
        model_out = model()

        params = get_layer_config(config, "changeLoss")
        extra_loss_layers.append(ChangeLoss(
            device=device,
            dtype=dtype,
            compare_pose=model_out.body_pose,
            **params))

    return extra_loss_layers
