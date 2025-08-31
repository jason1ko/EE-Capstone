import torch


def load_trained_ckd(model, saved_model_path):
    model_dict = model.state_dict()
    trained_dict = torch.load(saved_model_path, map_location='cpu')['state_dict']
    new_trained_dict = {}
    for k in trained_dict:
        if 'network' not in k:
            _k = f"network.{k}"
        else:
            _k = k
        new_trained_dict[_k] = trained_dict[k]  # tradition training

    model_dict.update(new_trained_dict)
    model.load_state_dict(model_dict)