import torch

VOCAB_SIZE = 3000
BLOCK_SIZE = 256

NUM_EMBD = 384
NUM_HEADS = 8
NUM_LAYERS = 8

BATCH_SIZE = 32
NUM_WORKERS = 0
MAX_ITERS = 12000
LEARNING_RATE = 3e-4
DROPOUT = 0.1
OUTPUT_DIR = "./data/models/"

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')