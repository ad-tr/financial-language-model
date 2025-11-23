import sentencepiece as spm
from model_transformer import GPT
from pathlib import Path
import config
import torch

def generate_text():
    print("--- Génération de texte ---")
    
    sp = spm.SentencePieceProcessor()
    sp.load('./data/tokenizer/tokenizer.model')

    model = GPT(
        vocab_size=config.VOCAB_SIZE,
        n_embd=config.NUM_EMBD,
        max_seq_len=config.BLOCK_SIZE,
        n_head=config.NUM_HEADS,
        dropout=config.DROPOUT,
        n_layer=config.NUM_LAYERS
    )
    
    save_path = Path(config.OUTPUT_DIR)
    final_name = save_path / "mon_transformer_final.pth"
    
    try:
        model.load_state_dict(torch.load(final_name, map_location=config.DEVICE))
        model.to(config.DEVICE)
        print("Poids chargés avec succès.")
    except FileNotFoundError:
        print("Erreur : Le fichier 'mon_transformer_final.pth' n'existe pas. Entraînez d'abord !")
        return

    model.eval()

    start_text = input("Entrez le debut du texte:\n")
    start_ids = sp.encode(start_text, out_type=int)
    context = torch.tensor([start_ids], dtype=torch.long, device=config.DEVICE) # (1, T)

    print(f"\nPrompt: {start_text}")
    print("Génération en cours...\n")
    
    generated_ids = model.generate(context, max_new_tokens=200)
    output_text = sp.decode(generated_ids[0].tolist())
    
    print("--- Résultat ---")
    print(output_text)

if __name__ == "__main__":
    generate_text()