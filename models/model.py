import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer


# There are 4 modules in the Image Caption model:
# ImageEncoder, TextDecoder, Mapping Module, Caption Net

# The ImageEncoder Module, return img embedding
class ImageEncoder(nn.Module):

    def __init__(self, model, device):
        super(ImageEncoder, self).__init__()
        self.preprocessor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(device)

    def forward(self, image, device):
        image = self.preprocessor(images=image, return_tensors='pt').to(device)
        image_features = self.model(**image)
        return image_features.pooler_output


# The TextDecoder Module, process embedding into caption
class TextDecoder(nn.Module):

    def __init__(self, model, device):
        super(TextDecoder, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model).to(device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embedding, attention_mask=None):
        text_features = self.model(inputs_embeds=embedding, attention_mask=attention_mask)
        return text_features.logits


# The Mapping Module, maps img to GPT2 embedding
class Mapping(nn.Module):

    def __init__(self, embed_size, device):
        super(Mapping, self).__init__()
        self.embed_size = embed_size
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=16, dim_feedforward=embed_size * 4, dropout=0.08,
                                       batch_first=True, device=device), num_layers=5).to(device)
        self.mapper = nn.Linear(embed_size, 4 * embed_size).to(device)
        self.init_weights()

    def forward(self, img_embedded, train_mode=False):
        x = self.transformer_encoder(img_embedded)
        x = self.mapper(x)
        x = x.view(*([-1, 4, self.embed_size] if train_mode else [4, self.embed_size]))
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# Caption Net, Concat all embedings together
class Caption_Net(nn.Module):

    def __init__(self, clip_model, text_model, max_len, device):
        super(Caption_Net, self).__init__()
        self.ep_len = 4
        self.ie = ImageEncoder(model=clip_model, device=device)
        self.mp = Mapping(embed_size=self.ie.model.config.hidden_size, device=device)
        self.td = TextDecoder(model=text_model, device=device)
        self.max_len = max_len
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.td.tokenizer.pad_token_id)
        self.freeze_layers()

    # freeze everything, except 1st and last transformer layer in Decoder
    def freeze_layers(self):
        for p in [*list(self.ie.parameters()), *list(self.td.parameters())[14:-14]]:
            p.requires_grad = False

    # for training
    def train_forward(self, img_emb, trg_cap, att_mask, device):
        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]
        img_mapped = self.mp(img_emb, train_mode=True)
        text_emb = self.td.model.transformer.wte(x)
        x = torch.concat([img_mapped, text_emb], dim=1)
        x_mask = torch.concat([torch.ones(x_mask.shape[0], self.ep_len).to(device), x_mask], dim=1)
        pos_emb = self.td.model.transformer.wpe(torch.arange(x.shape[1]).to(device))
        pos_emb = pos_emb.expand_as(x)
        x += pos_emb
        res = self.td(x, attention_mask=x_mask)
        res = torch.softmax(res, dim=2)
        loss = self.criterion(res[:, self.ep_len:, :].reshape(-1, res.shape[-1]), y.reshape(-1))
        return loss

    # for predict
    def forward(self, img, device):
        with torch.no_grad():
            img_embedded = self.ie(img, device)
            img_mapped = self.mp(img_embedded)
            sos_emb = self.td.model.transformer.wte(
                torch.tensor(self.td.tokenizer.bos_token_id).to(device))
            sos_emb = sos_emb.unsqueeze(0)
            start_emb = torch.cat([sos_emb, img_mapped], dim=0)

            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.td.model.transformer.wte(torch.tensor(tokens).to(device))
                    emb = torch.cat([start_emb, tok_emb], dim=0)
                else:
                    emb = start_emb
                pos_emb = self.td.model.transformer.wpe(torch.arange(emb.shape[0]).to(device))
                emb += pos_emb
                pred = self.td(emb)
                pred = torch.softmax(pred, dim=-1)
                _, pred = torch.max(pred, dim=1)
                last_token = pred[-1].item()
                tokens.append(last_token)
                if last_token == self.td.tokenizer.eos_token_id:
                    break
            decoded = self.td.tokenizer.decode(tokens[:-1])
            decoded = decoded.strip()

            if decoded:
                decoded = decoded[0].upper() + decoded[1:]
            else:
                decoded = ""
            return decoded
