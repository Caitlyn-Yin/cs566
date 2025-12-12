import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import re

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# IMG_HEIGHT = 96
# IMG_WIDTH = 512
# ENCODER_DIM = 512  # Default for ResNet34
# EMBEDDING_DIM = 256
# DECODER_HIDDEN_DIM = 512
# ATTENTION_DIM = 512
VOCAB_SIZE = 8000 # Placeholder, will be updated from checkpoint if possible or needs to match training

# Special Tokens (Must match training)
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

# --- Model Definitions (Copied/Imported from main.py structure) ---
# We need the class definitions to load the state dict. 
# Ideally, we would import these from main.py, but for a standalone demo script, 
# we'll redefine or import if main.py is importable. 
# Let's try to import from main to ensure consistency.

try:
    # from main import (
    #     CNNEncoder, ResNetEncoder, ViTEncoder,
    #     AttentionDecoder, LSTMDecoder, TransformerDecoder,
    #     Im2LatexModel, ResizeAndPad, 
    #     Tokenizer, normalize_latex
    # )
    from main import (
        EMBEDDING_DIM, DECODER_HIDDEN_DIM, ATTENTION_DIM, DEVICE, MAX_SEQ_LEN,
        CNNEncoder, ResNetEncoder, ViTEncoder,
        AttentionDecoder, LSTMDecoder, TransformerDecoder,
        Im2LatexModel, Tokenizer, ResizeAndPad,
        get_image_size, normalize_latex, beam_search_decode
    )
    # print("Successfully imported model classes from main.py")
except ImportError:
    print("Could not import from main.py. Please ensure main.py is in the same directory.")
    exit(1)

# --- Helper Functions ---

def load_model(model_path):
    if not model_path:
        # Default model path - try to find a .pth file in current dir
        pt_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if "im2latex_best_model_ResNetEncoder-AttentionDecoder.pth" in pt_files:
            model_path = "im2latex_best_model_ResNetEncoder-AttentionDecoder.pth"
            print(f"No path provided. Using default model: {model_path}")
        elif pt_files:
            model_path = pt_files[0]
            print(f"No path provided. Using found model: {model_path}")
        else:
            print("No model path provided and no .pth files found.")
            return None, None

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # We need to instantiate the correct model architecture.
    # The checkpoint might be just state_dict, or full model.
    # Assuming state_dict based on main.py
    
    # Heuristic to guess architecture from filename or try to load
    # If filename contains "resnet", use ResNetEncoder, else CNNEncoder
    # If filename contains "attention", use AttentionDecoder, else LSTMDecoder
    
    if "resnetencoder" in model_path.lower():
        encoder = ResNetEncoder(encoded_image_size=16).to(DEVICE)
        encoder_dim = 512 # ResNet34
    elif "vitencoder" in model_path.lower():
        encoder = ViTEncoder().to(DEVICE)
        encoder_dim = 768 # ViT-B/16
    else:
        encoder = CNNEncoder(encoded_image_size=16).to(DEVICE)
        encoder_dim = 512 # CNNEncoder output dim from main.py
        
    # We need to know vocab size to initialize decoder. 
    # In a real scenario, we should save vocab/tokenizer with the model.
    # For this demo, we'll try to load the tokenizer if available, or use a default size
    # and hope it matches.
    
    # Check if tokenizer is saved
    # If not, we might have issues decoding correctly if indices don't match.
    # Let's assume we can rebuild the tokenizer from the training data or it's pickled.
    # For now, let's try to load the training data to build the tokenizer just like main.py
    # This is slow but ensures correctness.
    
    # Load tokenizer
    TRAIN_CSV_PATH = './im2latex100k/im2latex_train.csv'
    VOCAB_PATH = './im2latex100k/vocab.json'
    # VOCAB_PATH = './im2latex100k/vocab_old.json'
    tokenizer = Tokenizer(min_freq=5)

    if os.path.exists(VOCAB_PATH):
        print(f"Loading vocabulary from {VOCAB_PATH}...")
        tokenizer.load(VOCAB_PATH)
        vocab_size = tokenizer.vocab_size
    elif os.path.exists(TRAIN_CSV_PATH):
        print("Rebuilding tokenizer from CSV (vocab.json not found)...")
        import pandas as pd
        df = pd.read_csv(TRAIN_CSV_PATH)
        corpus = [normalize_latex(f) for f in df['formula']]
        tokenizer.fit(corpus)
        vocab_size = tokenizer.vocab_size
    else:
        print("Warning: Training CSV and vocab.json not found. Using default vocab size (which might be wrong).")
        vocab_size = 8000 # Fallback
    
    if "attention" in model_path.lower():
        decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=encoder_dim,
            attention_dim=ATTENTION_DIM,
            dropout_p=0
        ).to(DEVICE)
    elif "lstm" in model_path.lower():
        decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=encoder_dim,
            dropout_p=0
        ).to(DEVICE)
    elif "transformer" in model_path.lower():
        # Transformer Parameters
        NUM_HEADS = 8
        NUM_LAYERS = 3
        # Note: DECODER_HIDDEN_DIM should be divisible by NUM_HEADS
        
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            decoder_hidden_dim=256, 
            encoder_dim=encoder_dim,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            max_len=MAX_SEQ_LEN,
            dropout_p=0
        ).to(DEVICE)
        

    model = Im2LatexModel(encoder, decoder).to(DEVICE)
    
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, tokenizer

def process_image(image_path, encoder_type):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    img = Image.open(image_path).convert('L')
    
    IMG_HEIGHT, IMG_WIDTH = get_image_size(encoder_type)
    # Transform
    transform = transforms.Compose([
        ResizeAndPad((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE) # Add batch dim
    return img_tensor

def predict(model, image_tensor, tokenizer):
    
    decoded_indices = beam_search_decode(model, image_tensor, beam_width=5, max_len=MAX_SEQ_LEN)
            
    # Convert indices to string
    latex_str = tokenizer.inverse_transform(decoded_indices[0])
    return latex_str

def main():
    print("--- Im2Latex Demo ---")
    
    # 1. Load Model
    model_path = input("Enter path to model checkpoint (leave empty for default): ").strip(' "\'')
    model, tokenizer = load_model(model_path)
    assert model is not None, "Failed to load model."
    encoder_type = model.encoder.__class__.__name__.lower()
    
    if model is None:
        print("Could not load model. Exiting.")
        return

    print("Model loaded successfully.")
    
    # 2. Loop for images
    while True:
        print("\n------------------------------------------------")
        image_path = input("Enter path to image file (or 'q' to quit): ").strip(' "\'')
        
        if image_path.lower() == 'q':
            break
            
        if not image_path:
            continue
            
        image_tensor = process_image(image_path, encoder_type)  
        if image_tensor is None:
            continue
            
        print("Processing image...")
        try:
            latex_output = predict(model, image_tensor, tokenizer)
            latex_output = normalize_latex(latex_output)
            # remove all spaces near []()\{\}^_
            latex_output = re.sub(r'\s*([\[\]\(\)\{\}\^_])\s*', r'\1', latex_output)
            # remove <space> token
            latex_output = latex_output.replace("<space>", " ")
            latex_output = latex_output.replace(r"\bf ", "")
            latex_output = re.sub(r'\s+', ' ', latex_output).strip()
            print("\nPredicted LaTeX:")
            print(f"{latex_output}")
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
