from models import __init__


name = 'Gaussian_Writing_GRU'
parameters={'n_gaussian' : 20, 'dropout':0, 'rnn_size':256}

model=get_model(name, parameters)

print(model)