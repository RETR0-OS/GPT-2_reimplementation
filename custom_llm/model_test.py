import torch
from components.tokenizer import BytePairTokenizer
from model import CustomLLM

def print_sample(model, prompt, ctx_len, tokenizer, device, max_new_tokens=100, temperature=0.3):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        output = model(input_ids[-ctx_len:])
        logits = output[:, -1, :] / temperature  # Apply temperature scaling
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    
    generated_text = tokenizer.decode(input_ids[0].tolist())
    print(generated_text)

if __name__ == "__main__":

    tokenizer = BytePairTokenizer("GPT_2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model  = CustomLLM(
        vocab_size=len(tokenizer),
        num_layers=6,  # Reduced for faster training
        embedding_dim=384,  # Reduced for faster training
        context_length=100,
        num_heads=6,
        dropout=0.1,
        skip_connections=True,
        kqv_bias=False
    ).to(device)
    model.load_state_dict(torch.load("trained_model.pt", map_location=device))
    # print(model)

    # Example usage
    print("Model loaded successfully.")
    print("Generating sample text:")
    print_sample(model, "HAMLET: To be or not to", 100, tokenizer, device, max_new_tokens=60) 
