# Install dependencies:
# pip install torch torchvision pandas numpy pillow nltk tqdm scikit-learn
import json
import os
import re
import random
from collections import Counter

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

TRAIN_PT_PATH = os.path.join(DATA_DIR, 'train_processed_space_token.pt')
VAL_PT_PATH = os.path.join(DATA_DIR, 'val_processed_space_token.pt')
TEST_PT_PATH = os.path.join(DATA_DIR, 'test_processed_space_token.pt')

# Model & Training Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 100
# LEARNING_RATE = 1e-3
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
CLIP_GRAD = 5.0
MAX_TEACHER_FORCING_RATIO = 1.0
MIN_TEACHER_FORCING_RATIO = 0.3
TF_ANNEAL_EPOCHS = EPOCHS             
SEED = 42
NUM_WORKERS = 4
# Image Parameters
IMG_HEIGHT = 96
IMG_WIDTH = 512 # Pad to a wide aspect ratio

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

def normalize_latex(formula):
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader

class Tokenizer:
    """Builds a vocabulary and handles token-to-index mapping."""
    def __init__(self, min_freq=5):
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
    then pads the rest with black.
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
        
        # Create a black canvas and paste the resized image
        new_img = Image.new("L", (self.output_width, self.output_height), 0)
        paste_x = (self.output_width - new_w) // 2
        paste_y = (self.output_height - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img

# Define image transformations
image_transform = transforms.Compose([
    ResizeAndPad((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(), # Converts to [0, 1] and (C, H, W)
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])

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
        # Pad to square, then resize to 224x224
        _, _, H, W = images.shape
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

class Im2LatexModel(nn.Module):
    def __init__(self, encoder, decoder, resume=False):
        super(Im2LatexModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.encoder_trainable = True

        checkpoint_path = f"im2latex_best_model_{self}_max-tfr-{MAX_TEACHER_FORCING_RATIO}_min-tfr-{MIN_TEACHER_FORCING_RATIO}.pth"
        if resume and os.path.exists(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print(f"Checkpoint loaded from {checkpoint_path}.")
        else:
            print("Starting training from scratch...")
    
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

# --- Training and Evaluation Loops ---

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

def evaluate(model, loader, criterion, tokenizer):
    model.eval()
    epoch_loss = 0
    
    references = [] # Ground truth (list of list of tokens)
    hypotheses = [] # Predictions (list of tokens)
    exact_match_count = 0
    
    with torch.no_grad():
        # pbar = tqdm(loader, desc="Evaluating")
        pbar = loader
        for images, targets in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass with no teacher forcing
            outputs = model(images, targets, teacher_forcing_ratio=0.0)
            
            # Calculate loss
            loss = criterion(
                outputs[:, 1:, :].reshape(-1, outputs.shape[2]),
                targets[:, 1:].reshape(-1)
            )
            epoch_loss += loss.item()
            
            # Decode predictions for metrics
            # Greedy decoding
            preds_idx = outputs.argmax(dim=2) # (B, max_len)
            
            for i in range(targets.shape[0]):
                pred_str = tokenizer.inverse_transform(preds_idx[i].cpu().numpy())
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

def test(encoder_type, decoder_type):
    """
    load the im2latex best model and evaluate on test set
    """
    print("Loading and preprocessing data for testing...")
    
    # Build tokenizer
    print("Building tokenizer...")
    VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.json')
    tokenizer = Tokenizer(min_freq=5)

    if os.path.exists(VOCAB_PATH):
        print(f"Loading vocabulary from {VOCAB_PATH}...")
        tokenizer.load(VOCAB_PATH)
        print(f"Vocabulary loaded. Total size: {tokenizer.vocab_size}")
    else:
        print("Error: Vocabulary not found. Please run training first.")
        return

    VOCAB_SIZE = tokenizer.vocab_size

    # Ensure test data is preprocessed
    if not os.path.exists(TEST_PT_PATH):
        print(f"Pre-processed Test data not found at {TEST_PT_PATH}.")
        print("Running one-time pre-processing for Test data...")
        df = load_data(TEST_CSV_PATH, IMG_DIR)
        temp_dataset = Im2LatexDataset(df, tokenizer, image_transform)
        temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
        processed_data = []
        for batch in tqdm(temp_loader, desc="Processing Test data"):
            image_tensor, formula_tensor = batch
            processed_data.append((image_tensor.squeeze(0).clone(), formula_tensor.squeeze(0).clone()))
        torch.save(processed_data, TEST_PT_PATH)
    
    # Load Test DataLoader
    test_dataset = PreprocessedIm2LatexDataset(TEST_PT_PATH)
    collate_fn = CollateFn(PAD_TOKEN)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print("Initializing model...")
    # Initialize models
    if encoder_type == "cnn_encoder":
        encoder = CNNEncoder(encoded_image_size=16).to(DEVICE)
    elif encoder_type == "resnet_encoder":
        encoder = ResNetEncoder(encoded_image_size=16).to(DEVICE)
    elif encoder_type == "vit_encoder":
        encoder = ViTEncoder().to(DEVICE)
    else:
        raise ValueError("Unsupported encoder type")

    # Determine encoder output dimension dynamically
    with torch.no_grad():
        dummy = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        enc_out = encoder(dummy)
    detected_encoder_dim = enc_out.shape[2]

    if decoder_type == "attention_decoder":
        decoder = AttentionDecoder(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=detected_encoder_dim,
            attention_dim=ATTENTION_DIM,
            dropout_p=DROPOUT
        ).to(DEVICE)
    elif decoder_type == "lstm_decoder":
        decoder = LSTMDecoder(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=detected_encoder_dim,
            dropout_p=DROPOUT
        ).to(DEVICE)
    else:
        raise ValueError("Unsupported decoder type")

    model = Im2LatexModel(encoder, decoder, resume=False).to(DEVICE)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Load best model weights
    checkpoint_path = f"im2latex_best_model_{model.module if isinstance(model, nn.DataParallel) else model}_max-tfr-{MAX_TEACHER_FORCING_RATIO}_min-tfr-{MIN_TEACHER_FORCING_RATIO}.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded best model from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    print("Starting evaluation on test set...")
    test_loss, test_bleu, test_em, test_ned = evaluate(
        model,
        test_loader,
        criterion,
        tokenizer
    )
    
    print(f"\n--- Test Set Evaluation ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test BLEU-4: {test_bleu:.4f}")
    print(f"Test Exact Match: {test_em:.4f}")
    print(f"Test NED: {test_ned:.4f}")

    results = {
        "test_loss": test_loss,
        "test_bleu": test_bleu,
        "test_em": test_em,
        "test_ned": test_ned
    }
    
    output_file = f"test_results_{model.module if isinstance(model, nn.DataParallel) else model}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

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
            for batch in tqdm(temp_loader, desc=f"Processing {name} data"):
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
        encoder_lr = 1e-3
    elif encoder_type == "resnet_encoder":
        encoder = ResNetEncoder(encoded_image_size=16).to(DEVICE)
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
        decoder_lr=1e-3
    elif decoder_type == "lstm_decoder":
        decoder = LSTMDecoder(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=detected_encoder_dim,
            dropout_p=DROPOUT
        ).to(DEVICE)
        decoder_lr=1e-3
    else:
        raise ValueError("Unsupported decoder type")

    model = Im2LatexModel(encoder, decoder, resume).to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    print(f"Model initialized: {model.module if isinstance(model, nn.DataParallel) else model}")
    
    # Optimizer and Loss
    optimizer_grouped_parameters = [
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr}
    ]
    
    optimizer = optim.Adam(optimizer_grouped_parameters, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    # Add a scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        # mode='max',  # Monitor val BLEU + EM - NED
        mode='min', # Monitor val loss
        factor=0.5,  
        patience=4,
        cooldown=2,
        min_lr=[5e-6, 5e-5]
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

        tf_decay = (MAX_TEACHER_FORCING_RATIO - MIN_TEACHER_FORCING_RATIO) / TF_ANNEAL_EPOCHS
        current_tf_ratio = get_teacher_forcing_ratio(
            epoch,
            MAX_TEACHER_FORCING_RATIO,
            MIN_TEACHER_FORCING_RATIO,
            TF_ANNEAL_EPOCHS
        )
        print(f"Using Teacher Forcing Ratio: {current_tf_ratio:.4f}")

        # if encoder_type in ["resnet_encoder", "vit_encoder"]:
        #     # Freeze encoder for first few epochs
        #     encoder_trainable = epoch > 5
        #     if not encoder_trainable:
        #         print("Freezing encoder parameters for this epoch.")
        # else:
        #     encoder_trainable = True
        
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            CLIP_GRAD,
            current_tf_ratio
            # encoder_trainable
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

        # scheduler.step(val_bleu + val_em - val_ned)  # Combine metrics for scheduling
        if epoch > 5:
            scheduler.step(val_loss)  # Use validation loss for scheduling
        
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
            torch.save(model_to_save.state_dict(), f"im2latex_best_model_{model_to_save}_max-tfr-{MAX_TEACHER_FORCING_RATIO}_min-tfr-{MIN_TEACHER_FORCING_RATIO}.pth")

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
    model_to_load.load_state_dict(torch.load(f"im2latex_best_model_{model_to_load}_max-tfr-{MAX_TEACHER_FORCING_RATIO}_min-tfr-{MIN_TEACHER_FORCING_RATIO}.pth", map_location=DEVICE))
    test_loss, test_bleu, test_em, test_ned = evaluate(
        model,
        test_loader,
        criterion,
        tokenizer
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

        # Save a checkpoint
        # if epoch % 10 == 0:
        #     torch.save(model.state_dict(), f"im2latex_baseline_epoch_{epoch}.pth")

    print("\nTraining complete.")
    # save training/validation/test records to a single json file
    records = {
        "train": train_record,
        "val": val_record,
        "test": test_record
    }
    with open(f"{model}_records_max-tfr-{MAX_TEACHER_FORCING_RATIO}_min-tfr-{MIN_TEACHER_FORCING_RATIO}.json", "w") as f:
        json.dump(records, f, indent=4)
    
if __name__ == "__main__":
    if not os.path.exists(IMG_DIR) or not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(VAL_CSV_PATH) or not os.path.exists(TEST_CSV_PATH):
        print(f"Dataset not found in '{DATA_DIR}'.")
        print("Please follow the prerequisite steps to download and unzip the dataset.")
    else:
        # main("resnet_encoder", "attention_decoder")
        # main("resnet_encoder", "lstm_decoder")
        # main("cnn_encoder", "lstm_decoder")
        # main("cnn_encoder", "attention_decoder")
        main("vit_encoder", "attention_decoder")
        main("vit_encoder", "lstm_decoder")
