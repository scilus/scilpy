from scilpy.ml.labelseg.labelsegnet import LabelSegNet

from dipy.utils.optpkg import optional_package

IMPORT_ERROR_MSG = "PyTorch is required to run this script. Please install" + \
                   " it first. See the official website for more info: " + \
                   "https://pytorch.org/get-started/locally/" # noqa
torch, have_torch, _ = optional_package('torch', trip_msg=IMPORT_ERROR_MSG)


def get_model(checkpoint_file, device):
    """ Get the model from a checkpoint. """

    # Load the model's hyper and actual params from a saved checkpoint
    try:
        checkpoint = torch.load(checkpoint_file, weights_only=False)
    except RuntimeError:
        # If the model was saved on a GPU and is being loaded on a CPU
        # we need to specify map_location=torch.device('cpu')
        checkpoint = torch.load(
            checkpoint_file, map_location=torch.device('cpu'),
            weights_only=False)

    state_dict = checkpoint['state_dict']
    hyperparameters = checkpoint['hyperparameters']
    print(hyperparameters)

    model = LabelSegNet(hyperparameters['in_chans'], hyperparameters['volume_size'], hyperparameters['prompt_strategy'], hyperparameters['embed_dim'], hyperparameters['bottleneck_dim'], n_bundles=hyperparameters['n_bundles'])

    model.load_state_dict(state_dict)
    model.to(device)

    # Put the model in eval mode to fix dropout and other stuff
    model.eval()

    return model
