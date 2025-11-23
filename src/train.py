from model_transformer import load_data, GPT, create_encode_corpus
import torch
import config
from pathlib import Path

def train():
    
    
    print("--- Lancement de l'entrainement ---")
    
    save_path = Path(config.OUTPUT_DIR)
    save_path.mkdir(parents=True, exist_ok=True) 
    print(f"Dossier de sauvegarde : {save_path.resolve()}")
    
    create_encode_corpus()
    print(f"Device defined to : {config.DEVICE}")
    
    train_loader, val_loader = load_data()
    
    model = GPT(
        vocab_size=config.VOCAB_SIZE,
        n_embd=config.NUM_EMBD,
        max_seq_len=config.BLOCK_SIZE,
        n_head=config.NUM_HEADS,
        dropout=config.DROPOUT,
        n_layer=config.NUM_LAYERS,
    )
    model.to(config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    for iter_num in range(config.MAX_ITERS):
        model.train()
        
        try:
            xb, yb = next(iter(train_loader))
        except StopIteration:
            train_loader, _ = load_data()
            xb, yb = next(iter(train_loader))
            
        xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter_num % 100 == 0:
            print(f"Step {iter_num}: Loss = {loss.item():.4f}")
            
        if iter_num > 0 and iter_num % 1000 == 0:
            checkpoint_name = save_path / f"checkpoint_{iter_num}.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"-> Modèle sauvegardé à l'étape {iter_num}")

    final_name = save_path / "mon_transformer_final.pth"
    torch.save(model.state_dict(), final_name)
    print("Modèle sauvegardé sous 'mon_transformer_final.pth'")

if __name__ == "__main__":
    train()