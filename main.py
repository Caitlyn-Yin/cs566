import json
import os
import re
import random
from collections import Counter
import math

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import models, transforms
from nltk.translate.bleu_score import corpus_bleu

import time
from nltk.metrics.distance import edit_distance

import getpass
import socket

username = getpass.getuser()
hostname = socket.gethostname()

print(f"{username}@{hostname}")
# --- Configuration ---

# Set data paths
DATA_DIR = './im2latex100k' # Directory where you unzipped the dataset
VOCAB_CSV_PATH = os.path.join(DATA_DIR, 'im2latex_formulas.csv')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'im2latex_train.csv')
VAL_CSV_PATH = os.path.join(DATA_DIR, 'im2latex_validate.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'im2latex_test.csv')
IMG_DIR = os.path.join(DATA_DIR, 'formula_images_processed')

TRAIN_PT_PATH = os.path.join(DATA_DIR, 'train_processed.pt')
VAL_PT_PATH = os.path.join(DATA_DIR, 'val_processed.pt')
TEST_PT_PATH = os.path.join(DATA_DIR, 'test_processed.pt')

# Model & Training Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 256
BATCH_SIZE = 512
EPOCHS = 100
# LEARNING_RATE = 1e-3
DROPOUT = 0.3
CLIP_GRAD = 5.0
LABEL_SMOOTHING = 0.1

MAX_SEQ_LEN = 180

TF_ANNEAL_EPOCHS = EPOCHS             
SEED = 42
NUM_WORKERS = 4
# Image Parameters

# Model Dimensions
EMBEDDING_DIM = 256
ATTENTION_DIM = 512
DECODER_HIDDEN_DIM = 512
# ENCODER_DIM = 2048 # ResNet-50 output feature dim
ENCODER_DIM = 512 # ResNet-34 output feature dim

# Special Tokens
PAD_TOKEN = 0
SOS_TOKEN = 1 # Start-of-Sequence
EOS_TOKEN = 2 # End-of-Sequence
UNK_TOKEN = 3 # Unknown

# Set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print(f"Using device: {DEVICE}")

# --- Data Preprocessing & Tokenizer ---

def normalize_latex_old(formula):
    if not isinstance(formula, str):
        return ""
    # replace two or more instances of "\," separated by some space  (example, \, \,  or \,  \, \,)with a space token <space>
    formula = re.sub(r'(\\,(\s*\\,)+)', ' <space> ', formula)
    # remove thin spaces \, and negative thin spaces \! only
    formula = re.sub(r'(\\[,!])', '', formula)
    # replace each instance of ~ \: \; \[space] \enspace \quad \qquad with a space token <space>
    formula = re.sub(r'(~|\\:|\\;|\\ |\\enspace|\\quad|\\qquad)', ' <space> ', formula)
    # replace \hspace { some text } \vspace { some text } with a space token <space>
    formula = re.sub(r'(\\hspace\s*{[^}]*}|\\vspace\s*{[^}]*})', ' <space> ', formula)
    # replace multiple instances of <space> separated by some space with a single <space>
    formula = re.sub(r'(<space>\s*)+', ' <space> ', formula)
    # replace multiple spaces with a single space
    formula = re.sub(r'\s+', ' ', formula).strip()
    assert "vspace" not in formula, f"vspace found in formula after normalization: {formula}"
    return formula


def normalize_latex(formula):
    fmla = formula
    if not isinstance(formula, str):
        return ""
    # replace two or more instances of "\," separated by some space  (example, \, \,  or \,  \, \,)with a space token <space>
    formula = re.sub(r'(\\,(\s*\\,)+)', ' <space> ', formula)
    # remove thin spaces \, and negative thin spaces \! only
    formula = re.sub(r'(\\[,!])', '', formula)
    # remove ule {} {}
    formula = re.sub(r'ule\s*\{[^}]*\}\s*\{[^}]*\}', '', formula)
    # remove \raisebox {}
    formula = re.sub(r'\\raisebox\s*\{[^}]*\}', '', formula)
    # replace each instance of ~ \: \; \[space] \enspace \quad \qquad with a space token <space>
    formula = re.sub(r'(~|\\:|\\;|\\ |\\enspace|\\quad|\\qquad|\\space|\\hfill|\\hfil(?![a-zA-Z]))', ' <space> ', formula)
    # replace \hspace { some text } \vspace { some text } with a space token <space>
    formula = re.sub(r'(\\hspace\s*\*?\s*\{[^\}]*\}|\\vspace\s*\*?\s*\{[^\}]*\})', ' <space> ', formula)
    # replace multiple instances of <space> separated by some space with a single <space>
    formula = re.sub(r'(<space>\s*)+', ' <space> ', formula)
    # replace \left< and right> with \langle and \rangle
    formula = re.sub(r'\\left<', r'\\langle', formula)
    formula = re.sub(r'\\right>', r'\\rangle', formula)
    # replace \prime with ', \lbrack with [, \rbrack with ], \lbrace with {, \rbrace with }, \vert with |
    formula = re.sub(r'\\prime', r"'", formula)
    formula = re.sub(r'\\lbrack', r'[', formula)
    formula = re.sub(r'\\rbrack', r']', formula)
    formula = re.sub(r'\\lbrace', r'{', formula)
    formula = re.sub(r'\\rbrace', r'}', formula)
    formula = re.sub(r'\\vert', r'|', formula)
    # replace \left. and \right. with nothing
    formula = re.sub(r'\\left\.|\\right\.', '', formula)
    # replace \left and \right with nothing
    formula = re.sub(r'\\left(?![a-zA-Z])|\\right(?![a-zA-Z])', '', formula)
    # replace \[bB]ig[g][lr] to nothing
    formula = re.sub(r'\\[bB]igg?[lrm]?(?![a-zA-Z])', '', formula)
    # replace \to with \rightarrow
    formula = re.sub(r'\\to(?![a-zA-Z])', r'\\rightarrow', formula)
    # replace \ne, \le, \ge with \neq, \leq, \geq
    formula = re.sub(r'\\ne(?![a-zA-Z])', r'\\neq', formula)
    formula = re.sub(r'\\le(?![a-zA-Z])', r'\\leq', formula)
    formula = re.sub(r'\\ge(?![a-zA-Z])', r'\\geq', formula)
    # replace \slash with /
    formula = re.sub(r'\\slash', r'/', formula)
    # replace \mathbf and \textbf with \bf;same for \it, \sf, \tt, \rm
    formula = re.sub(r'\\mathbf|\\textbf', r'\\bf', formula)
    formula = re.sub(r'\\mathit|\\textit', r'\\it', formula)
    formula = re.sub(r'\\mathsf|\\textsf', r'\\sf', formula)
    formula = re.sub(r'\\mathtt|\\texttt', r'\\tt', formula)
    formula = re.sub(r'\\mathrm|\\textrm', r'\\mathrm', formula) # \rm not in vocab
    # replace \triangle with \bigtriangleup
    formula = re.sub(r'\\triangle(?![a-zA-Z])', r'\\bigtriangleup', formula)
    # replace multiple spaces with a single space
    formula = re.sub(r'\s+', ' ', formula).strip()
    assert "vspace" not in formula, f"vspace found in formula after normalization: \nbefore: {fmla}\nafter: {formula}"
    return formula

def load_data(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    # Clean up image paths
    # Drop rows with missing images
    all_images = set(os.listdir(img_dir))
    df = df[df['image'].isin(all_images)].reset_index(drop=True)
    df['image'] = df['image'].apply(lambda x: os.path.join(img_dir, os.path.basename(x)))

    # Normalize LaTeX (basic cleanup)
    # df['formula'] = df['formula'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    print(f"{csv_path}: Loaded {len(df)} samples.")
    return df

def get_data_loaders(
    train_pt, 
    val_pt, 
    test_pt, 
    batch_size, 
    num_workers, 
    pad_idx
):
    
    # Create datasets
    train_dataset = PreprocessedIm2LatexDataset(train_pt)
    val_dataset = PreprocessedIm2LatexDataset(val_pt)
    test_dataset = PreprocessedIm2LatexDataset(test_pt)

    # Calculate sequence length statistics
    # for data, name in [(train_dataset.data, "Training"), (val_dataset.data, "Validation"), (test_dataset.data, "Test")]:
    #     seq_lengths = [len(item[1]) for item in data]
    #     seq_lengths.sort()
    #     total_samples = len(seq_lengths)
        
    #     print(f"\n--- Sequence Length Statistics ({name} Set) ---")
    #     percentiles = [0.50, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99, 0.999, 0.9999, 1.0]
    #     for p in percentiles:
    #         idx = int(p * total_samples) - 1
    #         val = seq_lengths[idx]
    #         print(f"{p*100}% percentile: {val}")
    #     print("---------------------------------------------\n")
    

    # # sample 25% of the training data for faster debugging. train_dataset.data is a list
    # subset_size = len(train_dataset) // 5
    
    # train_dataset.data = train_dataset.data[:subset_size]

    # Create collate function
    collate_fn = CollateFn(pad_idx)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # return None, None, 
    return train_loader, val_loader, test_loader

class Tokenizer:
    """Builds a vocabulary and handles token-to-index mapping."""
    def __init__(self, min_freq=10):
        self.word2idx = {"<pad>": PAD_TOKEN, "<sos>": SOS_TOKEN, "<eos>": EOS_TOKEN, "<unk>": UNK_TOKEN}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.min_freq = min_freq

    def fit(self, corpus):
        """Builds vocabulary from a list of normalized LaTeX strings."""
        words = " ".join(corpus).split()
        word_counts = Counter(words)
        
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        print(f"Vocabulary built. Total size: {self.vocab_size}")

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.word2idx, f, indent=4)

    def load(self, path):
        with open(path, 'r') as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    @property
    def vocab_size(self):
        return len(self.word2idx)

    def transform(self, formula_str):
        """Converts a string to a list of indices, including <sos> and <eos>."""
        tokens = formula_str.split()
        # max_tokens = MAX_SEQ_LEN - 2  # Reserve space for <sos> and <eos>
        # if len(tokens) > max_tokens:
        #     tokens = tokens[:max_tokens]
        indices = [self.word2idx.get(token, UNK_TOKEN) for token in tokens]
        return [SOS_TOKEN] + indices + [EOS_TOKEN]

    def inverse_transform(self, indices):
        """Converts a list of indices back to a string, stopping at <eos>."""
        words = []
        for idx in indices:
            if idx == EOS_TOKEN:
                break
            if idx not in [PAD_TOKEN, SOS_TOKEN]:
                words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

# --- Dataset and DataLoader ---

class ResizeAndPad:
    """
    Resizes an image to fit within (H, W) while maintaining aspect ratio,
    then pads the rest with white.
    """
    def __init__(self, output_size):
        self.output_height, self.output_width = output_size

    def __call__(self, img):
        # Calculate new size
        ratio = min(self.output_width / img.width, self.output_height / img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        
        # Use a resampling constant compatible with different Pillow versions
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img = img.resize((new_w, new_h), resample)
        
        # Create a white canvas and paste the resized image
        new_img = Image.new("L", (self.output_width, self.output_height), 255)
        paste_x = (self.output_width - new_w) // 2
        paste_y = (self.output_height - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img



class Im2LatexDataset(Dataset):
    def __init__(self, df, tokenizer, transform):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and transform image
        img_path = row['image']
        try:
            image = Image.open(img_path).convert('L') # Convert to grayscale
        except FileNotFoundError:
            print(f"Warning: Image not found {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        
        image = self.transform(image)
        
        # Normalize and tokenize formula
        try:
            formula = normalize_latex(row['formula'])
            tokens = self.tokenizer.transform(formula)
        except Exception as e:
            # Print detailed debug information to help trace which item/batch causes the error.
            print("\n--- DataLoader item error debug ---")
            print(f"Dataset index: {idx}")
            try:
                print(f"CSV row: {row.to_dict()}")
            except Exception:
                print(f"CSV row (repr): {repr(row)}")
            print(f"Image path: {img_path}")
            raw_formula = row.get('formula', None)
            print(f"Raw formula value: {repr(raw_formula)} (type: {type(raw_formula)})")
            print(f"normalize_latex output (if any): {repr(normalize_latex(raw_formula))}")
            print(f"Exception during tokenization: {e}")
            print("--- End debug ---\n", flush=True)
            # Re-raise so the DataLoader surface the original error after logging
            raise

        return image, torch.tensor(tokens).long()



# It loads pre-processed tensors from a list
class PreprocessedIm2LatexDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path)
        print(f"Loaded pre-processed data from {pt_path}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already a tuple of (image_tensor, formula_tensor)
        return self.data[idx]

class CollateFn:
    """Pads sequences in a batch to the same length."""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Filter out None values (from file-not-found errors)
        batch = [b for b in batch if b is not None]
        if not batch:
            return torch.tensor([]), torch.tensor([])

        images, formulas = zip(*batch)
        
        # Stack images (already same size)
        images = torch.stack(images, dim=0)
        
        # Pad formulas
        formulas = pad_sequence(formulas, batch_first=True, padding_value=self.pad_idx)
        
        return images, formulas

# --- Model Architecture 

class CNNEncoder(nn.Module):
    """
    CNN-based Encoder, without residual connections.
    """
    def __init__(self, encoded_image_size=16):
        super(CNNEncoder, self).__init__()
        self.encoded_image_size = encoded_image_size
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # (B, 64, H, W)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (B, 64, H/2, W/2)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (B, 128, H/2, W/2)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (B, 128, H/4, W/4)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (B, 256, H/4, W/4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # (B, 256, H/8, W/8)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # (B, 512, H/8, W/8)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 512, H/16, W/16)

            nn.Conv2d(512, ENCODER_DIM, kernel_size=3, stride=1, padding=1), # (B, encoder_dim, H/16, W/16)
            nn.BatchNorm2d(ENCODER_DIM),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (B, encoder_dim, H/32, W/32)
        )
    
    def __repr__(self):
        return "CNNEncoder"
    
    def __str__(self):
        return self.__repr__()
    
        
    def forward(self, images):
        out = self.conv_layers(images)

        # Permute and flatten: (B, 2048, H_enc, W_enc) -> (B, H_enc*W_enc, 2048)
        B, C, H_enc, W_enc = out.shape
        out = out.permute(0, 2, 3, 1) # (B, H_enc, W_enc, C)
        out = out.view(B, -1, C) # (B, num_pixels, encoder_dim)
        return out

class ResNetEncoder(nn.Module):
    """
    Encoder based on a pre-trained ResNet.
    Outputs a flattened sequence of features.
    """
    def __init__(self, encoded_image_size=16): # H/32 * W/32 = 3 * 16 = 48
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove the final fully connected and avgpool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # We need a 1-channel input (grayscale)
        # Modify the first conv layer
        self.resnet[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Adaptive pooling to get a fixed-size feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
    
    def __repr__(self):
        return "ResNetEncoder"
    
    def __str__(self):
        return self.__repr__()

    def forward(self, images):
        # images: (B, 1, H, W)
        out = self.resnet(images) # (B, 2048, H/32, W/32) e.g., (B, 2048, 3, 16)
        # out = self.adaptive_pool(out) # (B, 2048, H_enc, W_enc)
        
        # Permute and flatten: (B, 2048, H_enc, W_enc) -> (B, H_enc*W_enc, 2048)
        B, C, H_enc, W_enc = out.shape
        out = out.permute(0, 2, 3, 1) # (B, H_enc, W_enc, C)
        out = out.view(B, -1, C) # (B, num_pixels, encoder_dim)
        return out

class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder (ViT-B/16).
    Resizes input to 224x224 and adapts 1-channel input to 3-channel.
    """
    def __init__(self):
        super(ViTEncoder, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.hidden_dim = 768
        
    def forward(self, images):
        # images: (B, 1, H, W)
        _, _, H, W = images.shape
        if H != 224 or W != 224:
            # Pad to square, then resize to 224x224
            if H != W:
                size = max(H, W)
                # center the image
                padding = ((size - W) // 2, size - W - (size - W) // 2, (size - H) // 2, size - H - (size - H) // 2)
                images = F.pad(images, padding, mode='constant', value=1) # Pad with white (1)
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Repeat 1 channel to 3 channels
        images = images.repeat(1, 3, 1, 1)
        
        # Use torchvision internal processing
        x = self.vit._process_input(images)
        n = x.shape[0]
        
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        
        # Return sequence of patch embeddings (exclude class token)
        # Output shape: (B, num_patches, hidden_dim)
        return x[:, 1:]

    def __repr__(self):
        return "ViTEncoder"
    
    def __str__(self):
        return self.__repr__()
    
class Attention(nn.Module):
    """Bahdanau-style (additive) Attention."""
    def __init__(self, encoder_dim, decoder_hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.W_enc = nn.Linear(encoder_dim, attention_dim)
        self.W_dec = nn.Linear(decoder_hidden_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (B, num_pixels, encoder_dim)
        # decoder_hidden: (B, decoder_hidden_dim)
        
        proj_enc = self.W_enc(encoder_out) # (B, num_pixels, attention_dim)
        proj_dec = self.W_dec(decoder_hidden).unsqueeze(1) # (B, 1, attention_dim)
        
        energy = self.V(self.tanh(proj_enc + proj_dec)).squeeze(2) # (B, num_pixels)
        alpha = self.softmax(energy) # (B, num_pixels)
        
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # (B, encoder_dim)
        
        return context, alpha

class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, decoder_hidden_dim, encoder_dim, attention_dim, dropout_p):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.attention = Attention(encoder_dim, decoder_hidden_dim, attention_dim)
        
        # LSTMCell input is embedding + context vector
        self.lstm_cell = nn.LSTMCell(embedding_dim + encoder_dim, decoder_hidden_dim)
        
        # Linear layers to initialize hidden/cell states from mean encoder output
        self.init_h = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.tanh = nn.Tanh()
        
        # Final output layer
        self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)
        self.fc_dropout = nn.Dropout(dropout_p)
    
    def __repr__(self):
        return f"AttentionDecoder"
    
    def __str__(self):
        return self.__repr__()

    def init_hidden_state(self, encoder_out):
        """Initializes h and c states from the mean of encoder features."""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.tanh(self.init_h(mean_encoder_out)) # (B, decoder_hidden_dim)
        c = self.tanh(self.init_c(mean_encoder_out))
        return h, c

    def forward(self, encoder_out, targets, teacher_forcing_ratio=0.5):
        # encoder_out: (B, num_pixels, encoder_dim)
        # targets: (B, max_seq_len)
        
        B, max_seq_len = targets.shape
        
        # Initialize outputs tensor
        outputs = torch.zeros(B, max_seq_len, self.vocab_size).to(DEVICE)
        
        # Initialize hidden state
        h, c = self.init_hidden_state(encoder_out)
        
        # First input is <sos> token
        input_token = targets[:, 0] # (B,)
        
        for t in range(1, max_seq_len):
            # Embed the input token
            embedded = self.embedding(input_token) # (B, embedding_dim)
            embedded = self.embedding_dropout(embedded)
            
            # Get context vector from attention
            context, alpha = self.attention(encoder_out, h)
            
            # Concatenate embedding and context
            lstm_input = torch.cat((embedded, context), dim=1)
            
            # Run through LSTM cell
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Get prediction
            h_drop = self.fc_dropout(h)
            output = self.fc_out(h_drop) # (B, vocab_size)
            outputs[:, t, :] = output
            
            # Decide next input token
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = targets[:, t]
            else:
                input_token = output.argmax(1) # Greedy decoding
                
        return outputs


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, decoder_hidden_dim, encoder_dim, dropout_p):
        """
        A standard LSTM decoder without attention.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of token embeddings.
            decoder_hidden_dim: Dimension of the LSTM's hidden state.
            encoder_dim: Output dimension of the encoder (e.g., 2048 for ResNet-50).
            dropout_p: Dropout probability for the embedding layer and fc output layer.
        """
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_p)
        
        # LSTMCell input is now *only* the embedding dimension
        self.lstm_cell = nn.LSTMCell(embedding_dim, decoder_hidden_dim)
        
        # Linear layers to initialize hidden/cell states from the mean encoder output
        # This is how the encoder's context is passed to the decoder
        self.init_h = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.tanh = nn.Tanh()
        
        # Final output layer to map hidden state to vocab
        self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def __repr__(self):
        return f"LSTMDecoder"
    
    def __str__(self):
        return self.__repr__()

    def init_hidden_state(self, encoder_out):
        """
        Initializes h and c states from the mean of encoder features.
        encoder_out: (B, num_pixels, encoder_dim)
        """
        # Compress the spatial features into a single context vector
        mean_encoder_out = encoder_out.mean(dim=1) # (B, encoder_dim)
        
        # Project to the LSTM's hidden/cell dimension
        h = self.tanh(self.init_h(mean_encoder_out)) # (B, decoder_hidden_dim)
        c = self.tanh(self.init_c(mean_encoder_out))
        return h, c

    def forward(self, encoder_out, targets, teacher_forcing_ratio=0.5):
        """
        Forward pass.
        
        Args:
            encoder_out: (B, num_pixels, encoder_dim) - Output from the CNN encoder
            targets: (B, max_seq_len) - Ground truth target sequences
            teacher_forcing_ratio: Probability of using teacher forcing
        """
        
        B, max_seq_len = targets.shape
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(B, max_seq_len, self.vocab_size).to(DEVICE)
        
        # Initialize hidden state from the encoder output
        h, c = self.init_hidden_state(encoder_out)
        
        # First input to the decoder is the <sos> token
        input_token = targets[:, 0] # (B,)
        
        # Loop for each token in the sequence (t=1 is the first *prediction*)
        for t in range(1, max_seq_len):
            # Embed the input token
            embedded = self.embedding_dropout(self.embedding(input_token)) # (B, embedding_dim)

            # The input to the LSTM is *only* the embedding
            
            # Run one step of the LSTM
            h, c = self.lstm_cell(embedded, (h, c))
            
            # Get the output prediction
            h_drop = self.fc_dropout(h)
            output = self.fc_out(h_drop) # (B, vocab_size)
            outputs[:, t, :] = output
            
            # Decide the next input token
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                # Use the ground truth token
                input_token = targets[:, t]
            else:
                # Use the model's own prediction (greedy decoding)
                input_token = output.argmax(1) 
                
        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, Seq_Len, Dim)
        # pe: (Max_Len, Dim) -> slice to (Seq_Len, Dim) -> unsqueeze for batch
        x = x + self.pe[:x.size(1), :].unsqueeze(0) #type: ignore
        return self.dropout(x)
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, decoder_hidden_dim, encoder_dim, num_heads=8, num_layers=4, max_len=512, dropout_p=0.1):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.decoder_hidden_dim = decoder_hidden_dim
        
        # 1. Project Encoder Output if dimensions differ
        self.encoder_proj = nn.Linear(encoder_dim, decoder_hidden_dim)
        
        # 2. Embeddings and Positional Encoding
        # Note: In Transformers, we use decoder_hidden_dim as d_model
        self.embedding = nn.Embedding(vocab_size, decoder_hidden_dim)
        self.pos_encoder = PositionalEncoding(decoder_hidden_dim, dropout_p, max_len)
        
        # 3. Transformer Decoder Blocks
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=decoder_hidden_dim * 4, 
            dropout=dropout_p, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. Final Head
        self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        """Generates a triangular mask to prevent attending to future tokens."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, encoder_out, targets, teacher_forcing_ratio=1.0):
        """
        encoder_out: (B, num_pixels, encoder_dim)
        targets: (B, max_seq_len)
        """
        B, seq_len = targets.shape
        # truncate seq_len to max_len of positional encoding
        # if seq_len > self.pos_encoder.pe.size(0): #type: ignore
        #     seq_len = self.pos_encoder.pe.size(0) #type: ignore
        #     targets = targets[:, :seq_len]
        memory = self.encoder_proj(encoder_out) # (B, num_pixels, decoder_hidden_dim)

        # --- TRAINING MODE (Fast Parallel with Shifted Inputs) ---
        if teacher_forcing_ratio == 1.0:
            # IMPORTANT: Shift inputs right! 
            # Input:  [SOS, A, B, C]
            # Target: [A, B, C, EOS] (Handled by your loss function)
            tgt_inputs = targets[:, :-1] # Exclude the last token (EOS) for input
            
            
            # Embed the inputs
            tgt_emb = self.embedding(tgt_inputs)
            tgt_emb *= math.sqrt(self.decoder_hidden_dim)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # Create Causal Mask for the shifted length (seq_len - 1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_inputs.size(1), targets.device)
            
            # Pass through Transformer
            # output shape: (B, seq_len - 1, dim)
            trans_out = self.transformer_decoder(
                tgt=tgt_emb, 
                memory=memory, 
                tgt_mask=tgt_mask
            )
            
            # Project to Vocab: (B, seq_len - 1, vocab_size)
            logits = self.fc_out(trans_out)
            
            # --- PADDING FOR COMPATIBILITY ---
            # Your train_one_epoch loop expects output shape (B, seq_len, vocab_size)
            # and calculates loss on outputs[:, 1:].
            # So we pad the START of the sequence so our predictions align with targets[:, 1:]
            
            outputs = torch.zeros(B, seq_len, self.vocab_size).to(targets.device)
            outputs[:, 1:, :] = logits # Fill from index 1 onwards
            
            return outputs

        # --- INFERENCE MODE (Autoregressive Loop) ---
        else:
            # Start with <sos>
            generated_seq = targets[:, 0].unsqueeze(1) # (B, 1)
            outputs = torch.zeros(B, seq_len, self.vocab_size).to(targets.device)
            
            for t in range(seq_len):
                # Prepare current sequence
                tgt_emb = self.embedding(generated_seq)
                tgt_emb *= math.sqrt(self.decoder_hidden_dim)
                tgt_emb = self.pos_encoder(tgt_emb)
                tgt_mask = self.generate_square_subsequent_mask(generated_seq.size(1), targets.device)
                
                # Run Transformer
                transformer_out = self.transformer_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
                
                # Predict next token from the *last* position
                last_token_out = transformer_out[:, -1, :] # (B, dim)
                logits = self.fc_out(last_token_out)       # (B, vocab)
                outputs[:, t, :] = logits                  # Store prediction at current step
                
                # Greedy Decoding
                if t < seq_len - 1:
                    next_token = logits.argmax(1).unsqueeze(1)
                    generated_seq = torch.cat([generated_seq, next_token], dim=1)

            return outputs

    def __repr__(self):
        return "TransformerDecoder"

class Im2LatexModel(nn.Module):
    def __init__(self, encoder, decoder, resume=False, postfix=""):
        super(Im2LatexModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.encoder_trainable = True

        checkpoint_path = f"im2latex_best_model_{self}{postfix}.pth"
        if resume and os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print(f"Checkpoint loaded from {checkpoint_path}.")
        else:
            print("Initializing model from scratch...")
    
    def __repr__(self):
        return f"{self.encoder}-{self.decoder}"

    def __str__(self):
        return self.__repr__()
    
    def forward(self, images, targets, teacher_forcing_ratio=0.5):
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, targets, teacher_forcing_ratio)
        return outputs

    # def set_encoder_trainable(self, trainable):
    #     if self.encoder_trainable != trainable:
    #         self.encoder_trainable = trainable
    #         for param in self.encoder.parameters():
    #             param.requires_grad = trainable

    #     if trainable:
    #         self.encoder.train()
    #     else:
    #         self.encoder.eval()
        
    #     print(f"Encoder trainable set to {trainable}.")
SYNONYMS = {

}



# --- Training and Evaluation Loops ---

def beam_search_decode(mdl, images, beam_width=5, max_len=150):
    """
    Performs Beam Search for a BATCH of images using fully batched operations.
    Effective batch size becomes (B * beam_width).
    
    Args:
        model: The Im2LatexModel.
        images: Tensor of shape (B, C, H, W).
        beam_width: Number of beams to keep.
        max_len: Maximum sequence length.
        
    Returns:
        top_hypotheses: List of length B, where each element is the best hypothesis (list of token indices).
    """
    mdl.eval()
    with torch.no_grad():
        model = mdl.module if isinstance(mdl, nn.DataParallel) else mdl

        B = images.size(0)
        if beam_width == 1:
            # Create a zero target tensor with <sos> token
            zero_target = torch.full((B, max_len), PAD_TOKEN, device=DEVICE).long()
            zero_target[:, 0] = SOS_TOKEN  # Set the first token to <sos>
            
            # Forward pass with teacher forcing ratio = 0
            outputs = model(images, zero_target, teacher_forcing_ratio=0)
            
            # Get the predicted indices
            preds_idx = outputs.argmax(dim=2)  # (B, max_len)
            # Convert to list of lists
            return preds_idx.tolist()
        
        # 1. Encode images
        # encoder_out: (B, num_pixels, encoder_dim)
        encoder_out = model.encoder(images)
        vocab_size = model.decoder.vocab_size

    
        
        # --- Initialize Decoder State (Batch Size B) ---
        if isinstance(model.decoder, AttentionDecoder):
            h, c = model.decoder.init_hidden_state(encoder_out)
            state = (h, c)
        elif isinstance(model.decoder, LSTMDecoder):
            h, c = model.decoder.init_hidden_state(encoder_out)
            state = (h, c)
        elif isinstance(model.decoder, TransformerDecoder):
            # Transformer is stateless in the RNN sense, but needs memory
            memory = model.decoder.encoder_proj(encoder_out)
            state = memory # State is just the memory
        else:
            raise ValueError("Unknown decoder type")

        # --- First Step (t=0) ---
        # We start with B beams (1 per image). We will expand to B*k after the first step.
        start_token = torch.tensor([SOS_TOKEN] * B, device=DEVICE).long() # (B,)
        
        if isinstance(model.decoder, AttentionDecoder):
            h, c = state
            emb = model.decoder.embedding(start_token) # (B, emb_dim)
            context, alpha = model.decoder.attention(encoder_out, h)
            lstm_in = torch.cat((emb, context), dim=1)
            h_new, c_new = model.decoder.lstm_cell(lstm_in, (h, c))
            out = model.decoder.fc_out(h_new)
            log_probs = F.log_softmax(out, dim=1) # (B, V)
            state = (h_new, c_new)
            
        elif isinstance(model.decoder, LSTMDecoder):
            h, c = state
            emb = model.decoder.embedding(start_token)
            h_new, c_new = model.decoder.lstm_cell(emb, (h, c))
            out = model.decoder.fc_out(h_new)
            log_probs = F.log_softmax(out, dim=1)
            state = (h_new, c_new)
            
        elif isinstance(model.decoder, TransformerDecoder):
            # Input: [SOS] (B, 1)
            tgt = start_token.unsqueeze(1)
            tgt_emb = model.decoder.embedding(tgt) * math.sqrt(model.decoder.decoder_hidden_dim)
            tgt_emb = model.decoder.pos_encoder(tgt_emb)
            tgt_mask = model.decoder.generate_square_subsequent_mask(1, DEVICE)
            
            trans_out = model.decoder.transformer_decoder(tgt=tgt_emb, memory=state, tgt_mask=tgt_mask)
            # Last token output
            out = model.decoder.fc_out(trans_out[:, -1, :])
            log_probs = F.log_softmax(out, dim=1)
            # State (memory) remains same
            
        # --- Expand to Beam Width ---
        # Get top k candidates for each of the B images
        topk_log_probs, topk_indices = log_probs.topk(beam_width, dim=1) # (B, k)
        
        # Initialize Beam Scores: Flatten to (B*k,)
        beam_scores = topk_log_probs.view(-1) 
        
        # Initialize Sequences: (B*k, 1)
        seqs = topk_indices.view(-1, 1)
        
        # Expand Encoder Out: (B, ...) -> (B*k, ...)
        # We repeat each batch item k times: [Img1, Img1, ..., Img2, Img2, ...]
        encoder_out = encoder_out.unsqueeze(1).expand(B, beam_width, *encoder_out.shape[1:]).reshape(B * beam_width, *encoder_out.shape[1:])
        
        # Expand State
        if isinstance(state, tuple): # RNN
            h, c = state
            h = h.unsqueeze(1).expand(B, beam_width, -1).reshape(B * beam_width, -1)
            c = c.unsqueeze(1).expand(B, beam_width, -1).reshape(B * beam_width, -1)
            state = (h, c)
        else: # Transformer Memory
            state = state.unsqueeze(1).expand(B, beam_width, *state.shape[1:]).reshape(B * beam_width, *state.shape[1:])
            
        # --- Loop ---
        for step in range(1, max_len):
            # Input is the last token of each beam
            last_tokens = seqs[:, -1] # (B*k,)
            
            # Check for finished beams (EOS)
            is_finished = (last_tokens == EOS_TOKEN) # (B*k,)
            
            # If all beams are finished, we could stop, but some might be unfinished.
            # We continue until max_len.
            
            # --- Decoder Step (Batch Size B*k) ---
            if isinstance(model.decoder, AttentionDecoder):
                h, c = state
                emb = model.decoder.embedding(last_tokens)
                context, alpha = model.decoder.attention(encoder_out, h)
                lstm_in = torch.cat((emb, context), dim=1)
                h_new, c_new = model.decoder.lstm_cell(lstm_in, (h, c))
                out = model.decoder.fc_out(h_new)
                log_probs = F.log_softmax(out, dim=1)
                state_new = (h_new, c_new)
                
            elif isinstance(model.decoder, LSTMDecoder):
                h, c = state
                emb = model.decoder.embedding(last_tokens)
                h_new, c_new = model.decoder.lstm_cell(emb, (h, c))
                out = model.decoder.fc_out(h_new)
                log_probs = F.log_softmax(out, dim=1)
                state_new = (h_new, c_new)
                
            elif isinstance(model.decoder, TransformerDecoder):
                # Construct full input: [SOS] + seqs
                sos_tokens = torch.tensor([SOS_TOKEN] * (B * beam_width), device=DEVICE).unsqueeze(1)
                tgt = torch.cat([sos_tokens, seqs], dim=1) # (B*k, seq_len)
                
                tgt_emb = model.decoder.embedding(tgt) * math.sqrt(model.decoder.decoder_hidden_dim)
                tgt_emb = model.decoder.pos_encoder(tgt_emb)
                tgt_mask = model.decoder.generate_square_subsequent_mask(tgt.size(1), DEVICE)
                
                trans_out = model.decoder.transformer_decoder(tgt=tgt_emb, memory=state, tgt_mask=tgt_mask)
                out = model.decoder.fc_out(trans_out[:, -1, :])
                log_probs = F.log_softmax(out, dim=1)
                state_new = state # Memory unchanged
            
            # --- Handle Finished Beams ---
            # If a beam is finished, we force it to predict EOS (or PAD) with prob 1 (log_prob 0)
            # and everything else -inf. This keeps the score constant and prevents expansion.
            if is_finished.any():
                log_probs[is_finished, :] = -float('inf')
                log_probs[is_finished, EOS_TOKEN] = 0.0
            
            # --- Calculate Scores ---
            # beam_scores: (B*k,)
            # log_probs: (B*k, V)
            # Add current step log probs to accumulated scores
            next_scores = beam_scores.unsqueeze(1) + log_probs # (B*k, V)
            
            # Reshape to (B, k*V) to select top k across all extensions of the k beams for each image
            next_scores = next_scores.view(B, beam_width * vocab_size)
            
            # Top K
            topk_scores, topk_ids = next_scores.topk(beam_width, dim=1) # (B, k)
            
            # Decode IDs
            # topk_ids is index in [0, k*V - 1]
            prev_beam_indices = topk_ids // vocab_size # (B, k) [0..k-1]
            new_token_indices = topk_ids % vocab_size # (B, k)
            
            # Update Scores
            beam_scores = topk_scores.view(-1) # (B*k,)
            
            # --- Gather and Update State/Sequences ---
            # We need to map (b, k_idx) -> global_idx in (B*k)
            batch_offset = torch.arange(B, device=DEVICE).unsqueeze(1) * beam_width
            gather_indices = (batch_offset + prev_beam_indices).view(-1) # (B*k,)
            
            # Update State
            if isinstance(state_new, tuple):
                h, c = state_new
                h = h[gather_indices]
                c = c[gather_indices]
                state = (h, c)
            else:
                state = state_new[gather_indices]
                
            # Update Encoder Out (needed for Attention)
            encoder_out = encoder_out[gather_indices]
            
            # Update Sequences
            seqs = seqs[gather_indices] # Select the parent sequences
            new_tokens = new_token_indices.view(-1, 1)
            seqs = torch.cat([seqs, new_tokens], dim=1) # Append new tokens
            
        # --- Final Selection ---
        # The 0-th beam for each batch is the best because topk sorts by score (descending).
        best_indices = torch.arange(0, B * beam_width, beam_width, device=DEVICE)
        best_seqs = seqs[best_indices]
        
        # Convert to list of lists and prepend SOS
        best_seqs_list = best_seqs.tolist()
        final_output = []
        for s in best_seqs_list:
            final_output.append([SOS_TOKEN] + s)
            
        return final_output

def train_one_epoch(model, loader, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    real_model = model.module if isinstance(model, nn.DataParallel) else model
    # real_model.set_encoder_trainable(encoder_trainable)
    epoch_loss = 0
    
    # pbar = tqdm(loader, desc=f"Training Loss: {0:.4f}")
    pbar = loader
    for images, targets in pbar:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE) # (B, max_len)
        
        optimizer.zero_grad()
        
        # outputs: (B, max_len, vocab_size)
        outputs = model(images, targets, teacher_forcing_ratio)
        
        # Calculate loss
        # We need to ignore the <sos> token, so we compare
        # outputs[:, 1:] with targets[:, 1:]
        
        # Flatten for CrossEntropyLoss
        # outputs: (B * (max_len-1), vocab_size)
        # targets: (B * (max_len-1),)
        loss = criterion(
            outputs[:, 1:, :].reshape(-1, outputs.shape[2]),
            targets[:, 1:].reshape(-1)
        )
        
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        # pbar.set_description(f"Training Loss: {loss.item():.4f}")
    
        
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, tokenizer, beamer_width=1):
    model.eval()
    epoch_loss = 0
    
    references = [] # Ground truth (list of list of tokens)
    hypotheses = [] # Predictions (list of tokens)
    exact_match_count = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", maxinterval=60)
        # pbar = loader
        for images, targets in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            real_model = model.module if isinstance(model, nn.DataParallel) else model
            if isinstance(real_model.decoder, TransformerDecoder):
                if targets.shape[1] > MAX_SEQ_LEN:
                    targets = targets[:, :MAX_SEQ_LEN]
            
            # Forward pass with no teacher forcing
            outputs = model(images, targets, teacher_forcing_ratio=1.0)
            
            # Calculate loss
            loss = criterion(
                outputs[:, 1:, :].reshape(-1, outputs.shape[2]),
                targets[:, 1:].reshape(-1)
            )
            epoch_loss += loss.item()
            
            # Decode predictions for metrics
            # Greedy decoding
            # preds_idx = outputs.argmax(dim=2) # (B, max_len)
            preds_idx = beam_search_decode(model, images, beam_width=beamer_width, max_len=targets.shape[1])
            
            for i in range(targets.shape[0]):
                pred_str = tokenizer.inverse_transform(preds_idx[i])
                true_str = tokenizer.inverse_transform(targets[i].cpu().numpy())
                
                hypotheses.append(pred_str.split())
                references.append([true_str.split()])
                
                if pred_str == true_str:
                    exact_match_count += 1
                

                    
    # Calculate metrics
    val_loss = epoch_loss / len(loader)
    bleu_score = corpus_bleu(references, hypotheses)
    # Calculate NED (Normalized Edit Distance)
    ned_accum = 0
    for i in range(len(hypotheses)):
        hyp = hypotheses[i]
        ref = references[i][0] # references is a list of lists (for corpus_bleu)
        
        if len(hyp) == 0 and len(ref) == 0:
            ned_accum += 0
        else:
            # Token-level Levenshtein distance / max sequence length
            ned_accum += edit_distance(hyp, ref) / max(len(hyp), len(ref))
            
    ned_score = ned_accum / len(hypotheses) if hypotheses else 0.0
    # Ensure bleu_score is a float
    if isinstance(bleu_score, (list, tuple)):
        bleu_score = float(bleu_score[0]) if bleu_score else 0.0
    else:
        bleu_score = float(bleu_score)
    exact_match = exact_match_count / len(loader.dataset)
    
    return val_loss, bleu_score, exact_match, ned_score

def get_teacher_forcing_ratio(epoch, max_tfr, min_tfr, total_epochs):
    """Linearly decays teacher forcing ratio from max_tfr to min_tfr over total_epochs."""
    # Warmup for first 5% of epochs
    if epoch < int(0.05 * total_epochs):
        return max_tfr
    
    # Linear decay until 80% of epochs
    if epoch < int(0.8 * total_epochs):
        # Decaying from 1.0 to 0.3 over 70 epochs
        decay_rate = (max_tfr - min_tfr) / (int(0.8 * total_epochs) - int(0.05 * total_epochs))
        return max_tfr - (decay_rate * (epoch - int(0.05 * total_epochs)))
        
    # Floor at 0.3 for the rest
    return min_tfr

def get_image_size(encoder_type, decoder_type=None):
    if encoder_type in ["vit_encoder", "vitencoder"]:
        return (224, 224)
    else:
        return (96, 512)

def main(encoder_type, decoder_type, resume=False):
    print("Loading and preprocessing data...")
    # Load CSV
    # try:
    #     train_df = pd.read_csv(TRAIN_CSV_PATH)
    #     val_df = pd.read_csv(VAL_CSV_PATH) 
    #     test_df = pd.read_csv(TEST_CSV_PATH)
    # except FileNotFoundError:
    #     print(f"Error: Could not find '{TRAIN_CSV_PATH}', '{VAL_CSV_PATH}', or '{TEST_CSV_PATH}'.")
    #     print("Please download and unzip the 'im2latex100k' dataset first.")
    #     return

    # Filter out missing images from the CSV
    # all_images = set(os.listdir(IMG_DIR))
    # print(f"Total images found: {len(all_images)}")
    # train_df = train_df[train_df['image'].isin(all_images)].reset_index(drop=True)
    # test_df = test_df[test_df['image'].isin(all_images)].reset_index(drop=True)
    
    # Create a smaller subset for faster prototyping (e.g., 20k)
    # df = df.sample(n=20000, random_state=SEED).reset_index(drop=True)
    # print(f"Using a subset of {len(df)} samples for demonstration.")

    # Define image transformations
    IMG_HEIGHT, IMG_WIDTH = get_image_size(encoder_type, decoder_type)

    if encoder_type == "vit_encoder":
        global BATCH_SIZE
        BATCH_SIZE //= 2

    if decoder_type == "transformer_decoder":
        # For Transformers, use full teacher forcing during training
        MAX_TEACHER_FORCING_RATIO = 1.0
        MIN_TEACHER_FORCING_RATIO = 1.0
        WEIGHT_DECAY = 1e-4
    else:
        MAX_TEACHER_FORCING_RATIO = 1.0
        MIN_TEACHER_FORCING_RATIO = 0.3
        WEIGHT_DECAY = 1e-4

    
    image_transform = transforms.Compose([
        transforms.RandomApply([
        transforms.RandomAffine(
            degrees=1,              # Rotate between -1 and +1 degrees
            translate=(0.01, 0.01), # Shift vertically/horizontally by max 1%
            scale=(0.95, 1.05),     # Zoom between 95% and 105%
            shear=1                 # Shear (slant) by max 1 degree
        )
        ], p=0.5),
        ResizeAndPad((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(), # Converts to [0, 1] and (C, H, W)
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])
    
    # Build tokenizer
    print("Building tokenizer...")
    VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.json')
    tokenizer = Tokenizer(min_freq=5)

    if os.path.exists(VOCAB_PATH):
        print(f"Loading vocabulary from {VOCAB_PATH}...")
        tokenizer.load(VOCAB_PATH)
        print(f"Vocabulary loaded. Total size: {tokenizer.vocab_size}")
    else:
        vocab_df = pd.read_csv(VOCAB_CSV_PATH)
        corpus = [normalize_latex(f) for f in tqdm(vocab_df['formulas'])]
        tokenizer.fit(corpus)
        print(f"Saving vocabulary to {VOCAB_PATH}...")
        tokenizer.save(VOCAB_PATH)

    VOCAB_SIZE = tokenizer.vocab_size

    for (name, csv_path, pt_path) in [
        ("Training", TRAIN_CSV_PATH, TRAIN_PT_PATH),
        ("Validation", VAL_CSV_PATH, VAL_PT_PATH),
        ("Test", TEST_CSV_PATH, TEST_PT_PATH)
    ]:
        if not os.path.exists(pt_path):
            print(f"Pre-processed {name} data not found at {pt_path}.")
            print(f"Running one-time pre-processing for {name} data...")
            
            # Load the raw data paths/strings
            df = load_data(csv_path, IMG_DIR)
            print(f"{name} dataset size: {len(df)} samples.")
            
            # Use the on-the-fly dataset to do the processing
            temp_dataset = Im2LatexDataset(df, tokenizer, image_transform)

            temp_loader = DataLoader(
                temp_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=False # Not sending to GPU
            )
            
            processed_data = []
            # This loop is now just appending, while the workers
            # are doing the hard work in the background.
            for batch in tqdm(temp_loader, desc=f"Processing {name} data", mininterval=60):
                image_tensor, formula_tensor = batch
                # Squeeze the batch_size=1 dimension
                processed_data.append(
                    (image_tensor.squeeze(0).clone(), formula_tensor.squeeze(0).clone())
                )
            
            # Save the entire list of tensors
            torch.save(processed_data, pt_path)
            print(f"Saved pre-processed {name} data to {pt_path}.")
        else:
            print(f"Found pre-processed {name} data at {pt_path}.")
    

    # 4. Get DataLoaders
    # Now load directly from the pre-processed .pt files
    train_loader, val_loader, test_loader = get_data_loaders(
        TRAIN_PT_PATH,
        VAL_PT_PATH,
        TEST_PT_PATH,
        BATCH_SIZE,
        NUM_WORKERS,
        PAD_TOKEN
    )


    print("Initializing model...")
    # Initialize models
    if encoder_type == "cnn_encoder":
        encoder = CNNEncoder(encoded_image_size=16).to(DEVICE)
        if resume:
            encoder_lr = 1e-4
        else:
            encoder_lr = 1e-3
    elif encoder_type == "resnet_encoder":
        encoder = ResNetEncoder(encoded_image_size=16).to(DEVICE)
        if resume:
            encoder_lr = 1e-4
        else:
            encoder_lr = 1e-3
    elif encoder_type == "vit_encoder":
        encoder = ViTEncoder().to(DEVICE)
        encoder_lr = 1e-4
    else:
        raise ValueError("Unsupported encoder type")

    # Determine encoder output dimension dynamically
    with torch.no_grad():
        dummy = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        enc_out = encoder(dummy)
    detected_encoder_dim = enc_out.shape[2]
    print(f"Detected encoder output dimension: {detected_encoder_dim}")

    if decoder_type == "attention_decoder":
        decoder = AttentionDecoder(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=detected_encoder_dim,
            attention_dim=ATTENTION_DIM,
            dropout_p=DROPOUT
        ).to(DEVICE)
        if resume:
            decoder_lr=1e-4
        else:
            decoder_lr=1e-3
    elif decoder_type == "lstm_decoder":
        decoder = LSTMDecoder(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=detected_encoder_dim,
            dropout_p=DROPOUT
        ).to(DEVICE)
        if resume:
            decoder_lr=1e-4
        else:
            decoder_lr=1e-3
    elif decoder_type == "transformer_decoder":
        # Transformer Parameters
        # NUM_HEADS = 4
        # NUM_LAYERS = 3
        NUM_HEADS = 8
        NUM_LAYERS = 3
        # Note: DECODER_HIDDEN_DIM should be divisible by NUM_HEADS
        
        decoder = TransformerDecoder(
            vocab_size=VOCAB_SIZE,
            decoder_hidden_dim=256, 
            encoder_dim=detected_encoder_dim,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            max_len=MAX_SEQ_LEN,
            dropout_p=0.1
        ).to(DEVICE)
        if resume:
            decoder_lr = 5e-5
        else:
            decoder_lr = 1e-4
    else:
        raise ValueError("Unsupported decoder type")

    postfix = f"_max-tfr-{MAX_TEACHER_FORCING_RATIO}_min-tfr-{MIN_TEACHER_FORCING_RATIO}"

    model = Im2LatexModel(encoder, decoder, resume, postfix=postfix).to(DEVICE)

    checkpoint_path = f"im2latex_best_model_{model.module if isinstance(model, nn.DataParallel) else model}{postfix}.pth"

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    else:
        print("Using a single GPU or CPU.")

    print(f"Model initialized: {model.module if isinstance(model, nn.DataParallel) else model}")
    
    # Optimizer and Loss
    optimizer_grouped_parameters = [
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr}
    ]
    
    optimizer = optim.Adam(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=LABEL_SMOOTHING).to(DEVICE)

    # Add a scheduler
    if resume:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Monitor val BLEU + EM - NED
            # mode='min', # Monitor val loss
            factor=0.5,  
            patience=3,
            cooldown=1,
            min_lr=[1e-7, 1e-7]
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            # mode='max',  # Monitor val BLEU + EM - NED
            mode='min', # Monitor val loss
            factor=0.5,  
            patience=4,
            cooldown=2,
            min_lr=[1e-5, 1e-5]
        )
    print("Starting training...")
    best_bleu = 0.0
    best_em = 0.0
    best_ned = float('inf')
    epochs_no_improve = 0
    patience = 10  # Stop after 10 epochs with no improvement
    train_record = {"loss": []}
    val_record = {"loss": [], "bleu": [], "em": [], "ned": []}
    test_record = {"loss": [], "bleu": [], "em": [], "ned": []}
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        # if decoder_type == "transformer_decoder":
        if 0:
            if epoch <= 5:
                warm_lr = 1e-5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print(f"Warm start: Learning rate set to {warm_lr}")
            elif epoch == 6:
                optimizer.param_groups[0]['lr'] = encoder_lr
                optimizer.param_groups[1]['lr'] = decoder_lr
                print(f"Warm start finished. Restored learning rates: Encoder={encoder_lr}, Decoder={decoder_lr}")

        tf_decay = (MAX_TEACHER_FORCING_RATIO - MIN_TEACHER_FORCING_RATIO) / TF_ANNEAL_EPOCHS
        current_tf_ratio = get_teacher_forcing_ratio(
            epoch,
            MAX_TEACHER_FORCING_RATIO,
            MIN_TEACHER_FORCING_RATIO,
            TF_ANNEAL_EPOCHS
        )
        print(f"Using Teacher Forcing Ratio: {current_tf_ratio:.4f}")

       
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            CLIP_GRAD,
            current_tf_ratio
        )

        train_record["loss"].append(train_loss)
        
        val_loss, val_bleu, val_em, val_ned = evaluate(
            model,
            val_loader,
            criterion,
            tokenizer
        )

        val_record["loss"].append(val_loss)
        val_record["bleu"].append(val_bleu)
        val_record["em"].append(val_em)
        val_record["ned"].append(val_ned)


        # scheduler_start_epoch = 5 if decoder_type == "transformer_decoder" else 2
        scheduler_start_epoch = 2
        if epoch > scheduler_start_epoch:
            scheduler.step(val_bleu + val_em - val_ned)  # Combine metrics for scheduling
            # scheduler.step(val_loss)  # Use validation loss for scheduling
        
        # print date and time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] Epoch {epoch} Summary:")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tVal Loss:  {val_loss:.4f}")
        print(f"\tVal BLEU-4: {val_bleu:.4f}")
        print(f"\tVal Exact Match: {val_em:.4f}")
        print(f"\tVal NED: {val_ned:.4f}")

        if val_bleu > best_bleu or val_em > best_em or val_ned < best_ned:
            print(f"New best metric. Saving model...")
            best_bleu = max(best_bleu, val_bleu)
            best_em = max(best_em, val_em)
            best_ned = min(best_ned, val_ned)
            
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), checkpoint_path)

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Stopping early after {patience} epochs with no improvement.")
            break
    
    # load the best model for final evaluation
    print("Loading best model for final evaluation on test set...")
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_loss, test_bleu, test_em, test_ned = evaluate(
        model,
        test_loader,
        criterion,
        tokenizer,
        beamer_width=5
    )
    print(f"\n--- Test Set Evaluation ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test BLEU-4: {test_bleu:.4f}")
    print(f"Test Exact Match: {test_em:.4f}")
    print(f"Test NED: {test_ned:.4f}")
    test_record["loss"].append(test_loss)
    test_record["bleu"].append(test_bleu)
    test_record["em"].append(test_em)
    test_record["ned"].append(test_ned)


    print("\nTraining complete.")
    # save training/validation/test records to a single json file
    records = {
        "train": train_record,
        "val": val_record,
        "test": test_record
    }
    model_name = model.module if isinstance(model, nn.DataParallel) else model
    with open(f"{model_name}_records_max-tfr-{MAX_TEACHER_FORCING_RATIO}_min-tfr-{MIN_TEACHER_FORCING_RATIO}.json", "w") as f:
        json.dump(records, f, indent=4)
    
if __name__ == "__main__":
    if not os.path.exists(IMG_DIR) or not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(VAL_CSV_PATH) or not os.path.exists(TEST_CSV_PATH):
        print(f"Dataset not found in '{DATA_DIR}'.")
        print("Please follow the prerequisite steps to download and unzip the dataset.")
    else:
        main("resnet_encoder", "attention_decoder", resume=False)
        # main("resnet_encoder", "lstm_decoder", resume=True)
        # main("cnn_encoder", "lstm_decoder", resume=True)
        # main("cnn_encoder", "attention_decoder", resume=True)
        # main("vit_encoder", "attention_decoder", resume=True)
        # main("vit_encoder", "lstm_decoder", resume=True)
        main("resnet_encoder", "transformer_decoder", resume=False)