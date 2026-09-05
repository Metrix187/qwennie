"""Shared configuration for qwennie v2."""

T = 64
SLOTS = 8
P1 = 2 + SLOTS              # <u> + slots + <b>
A1 = 16
P2 = 2 + SLOTS
A2 = T - P1 - A1 - P2       # 28

TURN1_BOT = P1 - 1          # 9
TURN2_START = P1 + A1       # 26
TURN2_BOT = TURN2_START + P2 - 1  # 35
TURN2_GEN_START = TURN2_BOT + 1   # 36

# d=80 with five query heads gives a clean 16-dim head while MQA keeps
# the CSS KV cache tiny: 3 * 63 * 16 * K/V = 6,048 inherited properties.
D = 80
L = 3
Q_HEADS = 5
KV_HEADS = 1
HD = D // Q_HEADS
KV_DIM = KV_HEADS * HD
MLP = 160
LOCAL_WINDOW = 6
ROPE_BASE = 10000.0
EPS = 1e-5
MAX_VOCAB = 544

SEED = 17
STEPS = 1800
BATCH = 32
LR = 2.5e-3
MIN_LR = 2.5e-4
WD = 0.01
DROPOUT = 0.0

# 2:4 is applied during the final 40% of training and preserved at export.
SPARSITY_N = 2
SPARSITY_M = 4

# Exact hierarchical categorical sampler. Each group has at most this many tokens.
SAMPLE_GROUP = 16

SPECIALS = ["<p>", "<u>", "<b>", "<e>"]
PAD, USR, BOT, END = range(4)
