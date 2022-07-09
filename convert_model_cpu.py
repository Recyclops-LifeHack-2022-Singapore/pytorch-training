import torch
import os

MODEL_PATH = r'model_output\densenet_two_resize_99.model'

if __name__ == '__main__':
    # Load model
    model = torch.load(MODEL_PATH)
    model = model.to('cpu')

    save_filename = os.path.basename(MODEL_PATH).split('.')[0] + '_cpu'
    save_path = os.path.join(
        os.path.dirname(MODEL_PATH),
        save_filename + '.model'
    )
    torch.save(model, save_path)

