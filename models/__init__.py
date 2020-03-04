from models.Gaussian_Writing_GRU import Gaussian_Writing_GRU


def get_model(args):

    model_instance = _get_model_instance(args.arch)



    print('Fetching model %s - %s ' % (args.arch, args.model_name))

    if args.arch == 'Gaussian_Writing_GRU':

        model = model_instance(args.model_name, args.num_classes, args.input_channels, args.pretrained)

    else:

        raise 'Model {} not available'.format(args.arch)



    return model



def _get_model_instance(name):

    return {

        'Gaussian_Writing_GRU': Gaussian_Writing_GRU


    }[name]