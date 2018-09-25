from utils import unit
from keras.layers import Input
from keras import Model
import json
import itertools

FITLER = [128, 512]
CHANNEL = [32, 64, 128, 256, 512]
KERNAL = [3, 5, 7, 9, 11]


def save(model, config):
    for layer in model.layers:
        if layer.__class__.__name__ == 'Concatenate':
            continue

        layer_config = dict()
        layer_config['class_name'] = layer.__class__.__name__

        try:
            shape = layer.input_shape
        except Exception:
            shape = layer.get_input_shape_at(0)

        layer_config['input_shape'] = list(shape)[1:]
        layer_config['config'] = layer.get_config()
        config[layer.get_config()['name']] = layer_config


def create_model(kernal, filter, channel, dir):
    with open(dir + '/channel-config.json', 'w+') as fi:
        config = dict()
        X = Input([128, 128, channel])
        conv = unit.channel_unit(X, filter, (kernal, kernal), 'channel')
        model = Model(X, conv)
        save(model, config)
        json.dump(config, fi)

    with open(dir + '/filter-config.json', 'w+') as fi:
        config = dict()
        X = Input([128, 128, channel])
        conv = unit.filter_unit(X, filter, (kernal, kernal), 'filter')
        model = Model(X, conv)
        save(model, config)
        json.dump(config, fi)

    with open(dir + '/spatial-config.json', 'w+') as fi:
        config = dict()
        X = Input([128, 128, channel])
        conv = unit.xy_unit(X, filter, (kernal, kernal), 'spatial')
        model = Model(X, conv)
        save(model, config)
        json.dump(config, fi)


for k, f, c in itertools.product(KERNAL, FITLER, CHANNEL):
    create_model(k, f, c, 'resource/conv-config/{}/{}/{}'.format(k, f, c))
