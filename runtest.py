import os
import json
import torch
import torch.nn as nn
import argparse

from main import (
    DATA_DIR, TEST_PT_PATH, BATCH_SIZE, NUM_WORKERS, PAD_TOKEN, MAX_SEQ_LEN,
    EMBEDDING_DIM, DECODER_HIDDEN_DIM, ATTENTION_DIM, DROPOUT, DEVICE,
    Tokenizer, PreprocessedIm2LatexDataset, CollateFn,
    CNNEncoder, ResNetEncoder, ViTEncoder,
    AttentionDecoder, LSTMDecoder, TransformerDecoder,
    Im2LatexModel, evaluate, get_image_size
)

def test(encoder_type, decoder_type, checkpoint_path=None):
    print(f"Testing model: {encoder_type} + {decoder_type}")
    
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    vocab_path = os.path.join(DATA_DIR, 'vocab.json')
    tokenizer = Tokenizer(min_freq=5)
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
    else:
        print(f"Error: Vocab file not found at {vocab_path}")
        return
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # 2. Load Test Data
    print("Loading test data...")
    if not os.path.exists(TEST_PT_PATH):
        print(f"Error: Test data not found at {TEST_PT_PATH}. Please run main.py first to preprocess data.")
        return

    test_dataset = PreprocessedIm2LatexDataset(TEST_PT_PATH)
    collate_fn = CollateFn(PAD_TOKEN)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE // 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 3. Initialize Model
    print("Initializing model...")
    if encoder_type == "cnn":
        encoder = CNNEncoder(encoded_image_size=16).to(DEVICE)
    elif encoder_type == "resnet":
        encoder = ResNetEncoder(encoded_image_size=16).to(DEVICE)
    elif encoder_type == "vit":
        encoder = ViTEncoder().to(DEVICE)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    # Determine encoder output dimension dynamically
    IMG_HEIGHT, IMG_WIDTH = get_image_size(encoder_type, decoder_type)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
        enc_out = encoder(dummy)
    encoder_dim = enc_out.shape[2]
    print(f"Encoder output dimension: {encoder_dim}")

    if decoder_type == "attention":
        decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=encoder_dim,
            attention_dim=ATTENTION_DIM,
            dropout_p=0 # No dropout during evaluation
        ).to(DEVICE)
    elif decoder_type == "lstm":
        decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            decoder_hidden_dim=DECODER_HIDDEN_DIM,
            encoder_dim=encoder_dim,
            dropout_p=0
        ).to(DEVICE)
    elif decoder_type == "transformer":
         decoder = TransformerDecoder(
            vocab_size=vocab_size,
            decoder_hidden_dim=256, 
            encoder_dim=encoder_dim,
            num_heads=8, # Matching main.py default
            num_layers=3, # Matching main.py default
            max_len=MAX_SEQ_LEN,
            dropout_p=0
        ).to(DEVICE)
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")

    model = Im2LatexModel(encoder, decoder).to(DEVICE)

    # 4. Load Checkpoint
    if checkpoint_path is None:
        # Try to find a checkpoint matching the model architecture
        model_name = str(model) # e.g. ResNetEncoder-AttentionDecoder
        # Find files starting with im2latex_best_model_{model_name}
        files = [f for f in os.listdir('.') if f.startswith(f"im2latex_best_model_{model_name}") and f.endswith(".pth")]
        if files:
            # Sort by modification time to get the latest one, or just pick the first
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            checkpoint_path = files[0]
            print(f"No checkpoint specified. Found and using: {checkpoint_path}")
        else:
            print(f"Error: No checkpoint found for {model_name} in current directory.")
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Handle DataParallel wrapping
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    has_module = list(state_dict.keys())[0].startswith('module.')
    is_parallel = isinstance(model, nn.DataParallel)
    
    if has_module and not is_parallel:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    elif not has_module and is_parallel:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    # 5. Evaluate
    print("Running evaluation on test set...")
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    test_loss, test_bleu, test_em, test_ned = evaluate(
        model,
        test_loader,
        criterion,
        tokenizer,
        beamer_width=5
    )

    results = {
        "model": str(model.module if isinstance(model, nn.DataParallel) else model),
        "checkpoint": checkpoint_path,
        "test_loss": test_loss,
        "test_bleu": test_bleu,
        "test_em": test_em,
        "test_ned": test_ned
    }

    print("\n--- Test Results ---")
    print(json.dumps(results, indent=4))

    # 6. Save Results
    output_file = f"test_{encoder_type}_{decoder_type}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on the test set.")
    parser.add_argument("--encoder", "-en", type=str, required=True, help="cnn, resnet, vit")
    parser.add_argument("--decoder", "-de", type=str, required=True, help="lstm, attention, transformer")
    parser.add_argument("--checkpoint", "-cp", type=str, default=None, help="Path to checkpoint file (optional)")
    
    args = parser.parse_args()
    
    test(args.encoder, args.decoder, args.checkpoint)
