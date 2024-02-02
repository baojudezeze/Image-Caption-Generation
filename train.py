import argparse
import os
import pickle
import random
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import GPT2Tokenizer
import utils
from models.model import Caption_Net

# Cuda seed
random.seed(66)
np.random.seed(66)
torch.manual_seed(66)
torch.cuda.manual_seed(66)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # hyperparameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model.pt', help='model name')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='The rate for val dataset')
    parser.add_argument('--lr', type=int, default=5e-3)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get processed dataset
    try:
        with open(os.path.join('data', 'dataset.pkl'), 'rb') as f:
            data = pickle.load(f)
        dataset = data[:]
    except:
        utils.preprocessing_dataset(device)

    # GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    collate_fn_part = partial(utils.cl_fn, tokenizer=tokenizer)

    train_set, val_set = random_split(dataset, [1 - args.val_ratio, args.val_ratio])
    train_loader = DataLoader(train_set, batch_size=2 ** 5, collate_fn=collate_fn_part, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2 ** 5, collate_fn=collate_fn_part, shuffle=True)

    # model, optimizer and scheduler
    model = Caption_Net(clip_model="openai/clip-vit-large-patch14", text_model="gpt2-medium", max_len=50,
                        device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    warmup = utils.LRWarmup(epochs=args.epochs, max_lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup.lr_warmup)
    scaler = torch.cuda.amp.GradScaler()

    # build train model process with experiment tracking from wandb
    # wandb.login(key="b0d3489e2b123f172891f0397625b3ba38b4c201")
    # wandb.init(project='clipXgpt2 captioner')
    # wandb.watch(model, log='all')

    for epoch in range(args.epochs):

        # train
        model.train()
        epoch += 1
        total_loss = 0
        train_loss = []

        loop = tqdm(train_loader, total=len(train_loader))
        loop.set_description(f'Epoch: {epoch} | Loss: ---')
        for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):
            img_emb, cap, att_mask = img_emb.to(device), cap.to(device), att_mask.to(device)

            with torch.cuda.amp.autocast():
                loss = model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask, device=device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.3)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
            loop.set_description(f'Epoch: {epoch} | Loss: {total_loss / (batch_idx + 1):.3f}')
            loop.refresh()

        cur_lr = optimizer.param_groups[0]['lr']
        train_loss.append(total_loss / (batch_idx + 1))
        scheduler.step()

        # val
        model.eval()
        total_loss = 0
        valid_loss = []

        loop = tqdm(val_loader, total=len(val_loader))
        loop.set_description(f'Validation Loss: ---')
        for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):
            img_emb, cap, att_mask = img_emb.to(device), cap.to(device), att_mask.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    loss = model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask, device=device)
                    total_loss += loss.item()
                    loop.set_description(f'Validation Loss: {total_loss / (batch_idx + 1):.3f}')
                    loop.refresh()
        valid_loss.append(total_loss / (batch_idx + 1))

        # log loss to wandb
        # wandb.log({
        #     'train_loss/loss': train_loss,
        #     'valid_loss/loss': valid_loss,
        #     'lr': args.lr,
        # })

        # save model
        torch.save(model.state_dict(), "weights\model.pt")
