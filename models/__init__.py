from models.Gaussian_Writing_GRU import Gaussian_Writing_GRU


def get_model(name, parameters):

    if name == 'Gaussian_Writing_GRU':
        model = Gaussian_Writing_GRU(parameters["n_gaussian"], parameters["dropout"], parameters["rnn_size"])

    else:

        raise 'Model {} not available'.format(args.arch)



    return model


