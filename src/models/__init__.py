from .resnet34 import ResNet34
from .lenet import build_lenet
from .alexnet import build_alexnet
from .base3c3d import build_3c3d

def build_model(model_name):
    if model_name == 'ResNet34':
        model = ResNet34((28, 28, 1), 10)
        x, t, is_training, y_out, loss, accuracy = model.build([None, 32, 32, 3], [None, 10])
    elif model_name == 'LeNet':
        x, t, is_training, y_out, loss, accuracy = build_lenet([None, 32, 32, 3], [None, 10])
    elif model_name == 'AlexNet':
        x, t, is_training, y_out, loss, accuracy = build_alexnet([None, 32, 32, 3], [None, 10])
    elif model_name == '3c3d':
        x, t, is_training, y_out, loss, accuracy = build_3c3d([None, 32, 32, 3], [None, 10])

    return x, t, is_training, y_out, loss, accuracy