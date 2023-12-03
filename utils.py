import os
import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


# The flickr30k dataset is downloaded from kaggle,
# see https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
def preprocessing_dataset(device):
    # Load dataset
    df = pd.read_csv(os.path.join('raw', 'results.csv'), sep='|')
    df.columns = [col.strip() for col in df.columns]
    df = df.drop(['comment_number'], axis=1)
    ds = [(img_name, df[df['image_name'] == img_name]['comment'].values) for img_name, _ in df[0::5].to_numpy()]

    # Load CLIP model and processor
    preprocessor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').vision_model.to(device)

    # Based on loaded dataset, create a list of (image name, image embedding, caption) tuples
    results = []
    loop = tqdm(ds, total=len(ds), position=0, leave=True)
    for img_name, cap in loop:
        try:
            img = Image.open(os.path.join('data', 'raw', 'flickr30k_images', img_name))

            with torch.no_grad():
                img_prep = preprocessor(images=img, return_tensors='pt').to(device)
                img_features = model(**img_prep)
                img_features = img_features.pooler_output
                img_features = img_features.squeeze()
                img_features = img_features.cpu().numpy()
            for c in cap:
                results.append((img_name, img_features, c[1:]))
        except:
            print(f'Lack of image {img_name}')

    # save data into pickle file
    with open(os.path.join('data', 'dataset.pkl'), 'wb') as f:
        pickle.dump(results, f)


# collate_fn for DataLoader
def cl_fn(batch, tokenizer):
    batch = list(zip(*batch))
    _, img_emb, cap = batch
    del batch
    img_emb = torch.tensor(np.array(img_emb))
    cap = tokenizer(cap, padding=True, return_tensors='pt')
    input_ids, attention_mask = cap['input_ids'], cap['attention_mask']
    return img_emb, input_ids, attention_mask


# learning rate warm up
class LRWarmup():
    def __init__(self, epochs, max_lr):
        self.epochs = epochs
        self.max_lr = max_lr
        self.max_point = int(0.3 * self.epochs)

    def __call__(self, epoch):
        return self.lr_warmup(epoch)

    def lr_warmup(self, epoch):
        a_1 = self.max_lr / self.max_point
        a_2 = self.max_lr / (self.max_point - self.epochs)
        b = - a_2 * self.epochs
        return min(a_1 * epoch, a_2 * epoch + b)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessing_dataset(device)
