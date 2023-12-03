import os
import random

import numpy as np
import torch
from PIL import Image

from model import Caption_Net


def generate_caption(file_path):
    # Cuda seed
    random.seed(66)
    np.random.seed(66)
    torch.manual_seed(66)
    torch.cuda.manual_seed(66)
    torch.backends.cudnn.deterministic = True

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model and img path
    model_path = os.path.join('weights', 'model.pt')
    img = Image.open(file_path)

    # load model and return caption
    model = Caption_Net(clip_model="openai/clip-vit-large-patch14", text_model="gpt2-medium", max_len=50,
                        device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        caption = model(img, device)
    return '{}'.format(caption)
