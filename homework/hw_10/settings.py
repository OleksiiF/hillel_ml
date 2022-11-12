EPOCHS = {
    'name': 'num_epochs',
    'low': 1,
    'high': 30,
}
RANDOM_ROTATION = {
    'name': 'random rotation',
    'low': 5,
    'high': 15,
}
LR = {
    'name': 'lr',
    'low': 1e-5,
    'high': 1e-1,
    'step': 1e-5
}
KERNEL_SIZE = {
    'name': 'kernel size',
    'low': 2,
    'high': 4,
}
PADDING = {
    'name': 'padding',
    'low': 1,
    'high': 3,
}
STRIDE = {
    'name': 'stride',
    'low': 1,
    'high': 3,
}
OPTIMIZER = {
    "name": 'optimizer',
    'choices': [
        "Adam",
        "RMSprop",
        "SGD",
        "Adagrad",
        "Adadelta",
        "Adamax",
        "ASGD"
    ]
}
