import torch
from components.embedding_layer import EmbeddingLayer
from components.positional_emdedding import PositionalEmbedding
from components.transformer_block import TransformerBlock
from components.dataloader import create_data_loader
from components.tokenizer import BytePairTokenizer
from components.layer_normalizing import LayerNomalization
import time

class CustomLLM(torch.nn.Module):
    def __init__(self, vocab_size, num_layers=12, embedding_dim=768, context_length=32, num_heads=4, dropout=0.1, skip_connections=True, kqv_bias=False):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(context_length, embedding_dim)
        self.trasformer_blocks = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    TransformerBlock(embedding_dim, embedding_dim, context_length, num_heads, 0.1, kqv_bias=kqv_bias, skip_conn=skip_connections),
                    LayerNomalization(embedding_dim)
                )
                for _ in range(num_layers)
            ]
        )
        self.primary_dropout = torch.nn.Dropout(dropout)  # Dropout layer for regularization
        self.final_layer_norm = LayerNomalization(embedding_dim)
        self.output_layer = torch.nn.Linear(embedding_dim, vocab_size)  # Final linear layer to map to vocabulary size

    def forward(self, input_batch):
        embedded_input = self.embedding_layer(input_batch)
        positional_input = self.positional_embedding(embedded_input)
        positional_input = self.primary_dropout(positional_input)  # Apply dropout to positional embeddings
        layer_outs = positional_input
        layer_outs = self.trasformer_blocks(layer_outs)  # Pass through transformer blocks
        logits = self.output_layer(layer_outs)  # Final linear layer to project to vocabulary size
        return logits
    

def calc_batch_loss(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0.0
    if len(dataloader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_batch_loss(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches if num_batches > 0 else float('nan')

def print_sample(model, input_text, tokenizer, device, max_new_tokens=50, temperature=0.3, top_k=30):
    """Generate and print a sample text using the model."""
    model.eval()
    with torch.no_grad():
        context_tokens = torch.tensor(tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(device)
        
        for _ in range(max_new_tokens):
            # Get predictions
            logits = model(context_tokens)
            # Get the logits for the last position

            top_logits, pos_mask = torch.topk(logits[:, -1, :], k=top_k)
            
            out_tensor = torch.full_like(logits, fill_value=-torch.inf, device=device, requires_grad=False)
            out_tensor.scatter_(1, pos_mask, top_logits)

            logits = out_tensor / temperature
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            # Append to context
            context_tokens = torch.cat([context_tokens, next_token], dim=1)
        
        # Decode and print
        generated_text = tokenizer.decode(context_tokens.squeeze(0).tolist())
        print(f"Generated text: {generated_text}")

def pretrain_model(model, train_data, validation_data, device, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """
    Pretrain the model with proper training loop.
    
    Args:
        model: The model to train
        train_data: Training data loader
        validation_data: Validation data loader
        device: Device to train on
        optimizer: Optimizer for training
        num_epochs: Number of epochs to train
        eval_freq: Frequency of evaluation (every N epochs)
        eval_iter: Number of batches to use for evaluation
        start_context: Context string for text generation samples
        tokenizer: Tokenizer for decoding
    """
    train_losses, val_losses = [], []
    
    print(f"Starting pretraining for {num_epochs} epochs...")
    print(f"Training batches: {len(train_data)}")
    print(f"Validation batches: {len(validation_data)}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # Training loop
        for input_batch, target_batch in train_data:
            optimizer.zero_grad()
            loss = calc_batch_loss(model, input_batch, target_batch, device)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('nan')
        
        # Evaluation
        if (epoch + 1) % eval_freq == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_loss = calc_loss_loader(train_data, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(validation_data, model, device, num_batches=eval_iter)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Average Training Loss: {avg_epoch_loss:.4f}")
                print(f"  Eval Train Loss: {train_loss:.4f}")
                print(f"  Eval Val Loss: {val_loss:.4f}")
                print_sample(model, start_context, tokenizer, device)
                print("-" * 50)
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_epoch_loss:.4f}")
    
    return train_losses, val_losses


if __name__ == "__main__":
    # Load and prepare data
    with open("../data/hamlet.txt", "r", encoding="utf-8") as f:
        text = f.read()

    train_test_ratio = 0.9
    train_text = text[:int(len(text) * train_test_ratio)]
    test_text = text[int(len(text) * train_test_ratio):]

    # Initialize tokenizer
    tokenizer = BytePairTokenizer("GPT_2")
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Create data loaders
    context_length = 100
    batch_size = 4
    
    train_data = create_data_loader(
        batch_size=batch_size,
        text=train_text,
        tokenizer=tokenizer,
        context_length=context_length,
        stride=context_length // 2  # 50% overlap
    )

    validation_data = create_data_loader(
        batch_size=batch_size,
        text=test_text,
        tokenizer=tokenizer,
        context_length=context_length,
        stride=context_length // 2
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = CustomLLM(
        vocab_size=vocab_size,
        num_layers=6,  # Reduced for faster training
        embedding_dim=384,  # Reduced for faster training
        context_length=context_length,
        num_heads=6,
        dropout=0.1,
        skip_connections=True,
        kqv_bias=False
    )

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    # Training parameters
    num_epochs = 10
    eval_freq = 2  # Evaluate every 2 epochs
    eval_iter = 5   # Use 5 batches for evaluation
    start_context = "But let this same "

    # Calculate initial losses
    print("\nCalculating initial losses...")
    model.eval()
    with torch.no_grad():
        initial_train_loss = calc_loss_loader(train_data, model, device, num_batches=eval_iter)
        initial_val_loss = calc_loss_loader(validation_data, model, device, num_batches=eval_iter)
    
    print(f"Initial training loss: {initial_train_loss:.4f}")
    print(f"Initial validation loss: {initial_val_loss:.4f}")

    # Start pretraining
    print("\nStarting pretraining...")
    start_time = time.perf_counter()
    
    train_losses, val_losses = pretrain_model(
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        device=device,
        optimizer=optimizer,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=eval_iter,
        start_context=start_context,
        tokenizer=tokenizer
    )
    
    end_time = time.perf_counter()
    print(f"\nPretraining completed in {end_time - start_time:.2f} seconds")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pt")
    print("Model saved as 'trained_model.pt'")

    # Final evaluation
    print("\nFinal evaluation:")
    model.eval()
    with torch.no_grad():
        final_train_loss = calc_loss_loader(train_data, model, device, num_batches=None)
        final_val_loss = calc_loss_loader(validation_data, model, device, num_batches=None)
    
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Generate a longer sample
    # print("\nGenerating sample text:")
    # print_sample(model, "HAMLET: To be or not to be", tokenizer, device, max_new_tokens=100)