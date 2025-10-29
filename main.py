# Install dependencies:
# test upload
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

# --- Configuration ---

# Set data paths
DATA_DIR = './im2latex100k' # Directory where you unzipped the dataset
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'im2latex_train.csv')
VAL_CSV_PATH = os.path.join(DATA_DIR, 'im2latex_validate.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'im2latex_test.csv')
IMG_DIR = os.path.join(DATA_DIR, 'formula_images_processed')

# Model & Training Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
CLIP_GRAD = 5.0
TEACHER_FORCING_RATIO = 0.5
SEED = 42

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
    # """
    # Normalizes LaTeX strings to handle variations (your "side task").
    # - Removes delimiters like $
    # - Adds spaces around tokens
    # - Collapses multiple spaces
    # """
    # formula = re.sub(r"(\$|\\\(|\\\))", "", formula) # Remove $ and \( \)
    # formula = re.sub(r"\\( |mathrm|text|mathbf)\{.*?\}", "", formula) # Remove \text{} etc.
    # formula = re.sub(r"([\{\}\(\)\[\]_\\^])", r" \1 ", formula) # Add space around brackets, etc.
    # formula = re.sub(r"([+\-=<>,.!])", r" \1 ", formula) # Add space around operators
    # formula = re.sub(r"\\([a-zA-Z]+)", r" \\\1 ", formula) # Add space around commands
    # formula = re.sub(r"\s+", " ", formula).strip() # Collapse multiple spaces
    if not isinstance(formula, str):
        return ""
    return formula

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

class LatexDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer, transform):
        self.df = df
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and transform image
        img_path = os.path.join(self.img_dir, row['image'])
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

# --- Model Architecture (CNN Encoder + Attention Decoder) ---

class CNNEncoder(nn.Module):
    """
    Encoder based on a pre-trained ResNet.
    Outputs a flattened sequence of features.
    """
    def __init__(self, encoded_image_size=16): # H/32 * W/32 = 3 * 16 = 48
        super(CNNEncoder, self).__init__()
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

    def forward(self, images):
        # images: (B, 1, H, W)
        out = self.resnet(images) # (B, 2048, H/32, W/32) e.g., (B, 2048, 3, 16)
        # out = self.adaptive_pool(out) # (B, 2048, H_enc, W_enc)
        
        # Permute and flatten: (B, 2048, H_enc, W_enc) -> (B, H_enc*W_enc, 2048)
        B, C, H_enc, W_enc = out.shape
        out = out.permute(0, 2, 3, 1) # (B, H_enc, W_enc, C)
        out = out.view(B, -1, C) # (B, num_pixels, encoder_dim)
        return out

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
    def __init__(self, vocab_size, embedding_dim, decoder_hidden_dim, encoder_dim, attention_dim):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(encoder_dim, decoder_hidden_dim, attention_dim)
        
        # LSTMCell input is embedding + context vector
        self.lstm_cell = nn.LSTMCell(embedding_dim + encoder_dim, decoder_hidden_dim)
        
        # Linear layers to initialize hidden/cell states from mean encoder output
        self.init_h = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.tanh = nn.Tanh()
        
        # Final output layer
        self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)

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
            
            # Get context vector from attention
            context, alpha = self.attention(encoder_out, h)
            
            # Concatenate embedding and context
            lstm_input = torch.cat((embedded, context), dim=1)
            
            # Run through LSTM cell
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Get prediction
            output = self.fc_out(h) # (B, vocab_size)
            outputs[:, t, :] = output
            
            # Decide next input token
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = targets[:, t]
            else:
                input_token = output.argmax(1) # Greedy decoding
                
        return outputs


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, decoder_hidden_dim, encoder_dim):
        """
        A standard LSTM decoder without attention.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of token embeddings.
            decoder_hidden_dim: Dimension of the LSTM's hidden state.
            encoder_dim: Output dimension of the encoder (e.g., 2048 for ResNet-50).
        """
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTMCell input is now *only* the embedding dimension
        self.lstm_cell = nn.LSTMCell(embedding_dim, decoder_hidden_dim)
        
        # Linear layers to initialize hidden/cell states from the mean encoder output
        # This is how the encoder's context is passed to the decoder
        self.init_h = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_hidden_dim)
        self.tanh = nn.Tanh()
        
        # Final output layer to map hidden state to vocab
        self.fc_out = nn.Linear(decoder_hidden_dim, vocab_size)

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
            embedded = self.embedding(input_token) # (B, embedding_dim)

            # The input to the LSTM is *only* the embedding
            
            # Run one step of the LSTM
            h, c = self.lstm_cell(embedded, (h, c))
            
            # Get the output prediction
            output = self.fc_out(h) # (B, vocab_size)
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
    def __init__(self, encoder, decoder):
        super(Im2LatexModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, images, targets, teacher_forcing_ratio=0.5):
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, targets, teacher_forcing_ratio)
        return outputs

# --- Training and Evaluation Loops ---

def train_one_epoch(model, loader, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
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
    exact_match = exact_match_count / len(loader.dataset)
    
    return val_loss, bleu_score, exact_match

# --- Main Execution ---

def main():
    print("Loading and preprocessing data...")
    # Load CSV
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        val_df = pd.read_csv(VAL_CSV_PATH) # Not used in this baseline
        test_df = pd.read_csv(TEST_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find '{TRAIN_CSV_PATH}', '{VAL_CSV_PATH}', or '{TEST_CSV_PATH}'.")
        print("Please download and unzip the 'im2latex100k' dataset first.")
        return

    # Filter out missing images from the CSV
    all_images = set(os.listdir(IMG_DIR))
    print(f"Total images found: {len(all_images)}")
    train_df = train_df[train_df['image'].isin(all_images)].reset_index(drop=True)
    test_df = test_df[test_df['image'].isin(all_images)].reset_index(drop=True)
    
    # Create a smaller subset for faster prototyping (e.g., 20k)
    # df = df.sample(n=20000, random_state=SEED).reset_index(drop=True)
    # print(f"Using a subset of {len(df)} samples for demonstration.")
    
    # Build tokenizer
    print("Building tokenizer...")
    corpus = [normalize_latex(f) for f in train_df['formula']]
    print(len(corpus))
    tokenizer = Tokenizer(min_freq=5)
    VOCAB_SIZE = tokenizer.vocab_size
    
    
    # Create datasets
    train_dataset = LatexDataset(train_df, IMG_DIR, tokenizer, image_transform)
    test_dataset = LatexDataset(test_df, IMG_DIR, tokenizer, image_transform)
    
    # Create dataloaders
    collate_fn = CollateFn(pad_idx=PAD_TOKEN)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print("Initializing model...")
    # Initialize models
    encoder = CNNEncoder(encoded_image_size=16).to(DEVICE)
    # decoder = AttentionDecoder(
    #     vocab_size=VOCAB_SIZE,
    #     embedding_dim=EMBEDDING_DIM,
    #     decoder_hidden_dim=DECODER_HIDDEN_DIM,
    #     encoder_dim=ENCODER_DIM,
    #     attention_dim=ATTENTION_DIM
    # ).to(DEVICE)

    decoder = LSTMDecoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
        encoder_dim=ENCODER_DIM
    ).to(DEVICE)
    
    model = Im2LatexModel(encoder, decoder).to(DEVICE)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            CLIP_GRAD,
            TEACHER_FORCING_RATIO
        )
        
        val_loss, bleu, em = evaluate(
            model,
            test_loader,
            criterion,
            tokenizer
        )
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tTest Loss:  {val_loss:.4f}")
        print(f"\tTest BLEU-4: {bleu:.4f}")
        print(f"\tTest Exact Match: {em:.4f}")

        # Save a checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"im2latex_baseline_epoch_{epoch}.pth")

    print("\nTraining complete.")
    
if __name__ == "__main__":
    if not os.path.exists(IMG_DIR) or not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(VAL_CSV_PATH) or not os.path.exists(TEST_CSV_PATH):
        print(f"Dataset not found in '{DATA_DIR}'.")
        print("Please follow the prerequisite steps to download and unzip the dataset.")
    else:
        main()
