"""Single source of truth for every setting. Edit by hand; nothing overrides these."""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# Fields that define the loaded model replicas (loaded once per sweep) -> not sweepable.
REPLICA_FIELDS = ("model_id", "dtype", "devices", "cache_dir")

# hi :)

# Funnel topology: agents per layer (source first); ids are consecutive by layer. THE place to
# edit it: n_agents, custom_edges and layer_sizes below all derive from it.
LAYER_SIZES = (1, 8, 6, 4, 2, 1)


def layered_edges(sizes):
    """Directed edges, full bipartite from each layer to the next (forward only)."""
    starts = [sum(sizes[:k]) for k in range(len(sizes))]
    return tuple(
        (src, dst)
        for k in range(len(sizes) - 1)
        for src in range(starts[k], starts[k] + sizes[k])
        for dst in range(starts[k + 1], starts[k + 1] + sizes[k + 1])
    )

@dataclass(frozen=True)
class Config:
    # Sweep: tuple of (field_name, values) pairs; any Config field except REPLICA_FIELDS and
    # `sweep` itself. One run per cartesian-product combo; () -> a single run with this config.
    # Format: (("alpha", (1, 2, 4)), ("persona_trait", ("outgoing", "solitary")))
    # steering_mode 'none' ignores alpha, so it is run once per remaining combo (see swarm.py).
    sweep: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (
        ("steering_mode", ("persona", "random",)),
        ("alpha", (-4, -0.5, 0.1, 2, 8)),
        ("persona_trait", ("compassionate", "self-interested")), # "consistent", "careless", "outgoing", "solitary", 
    )  # swept values replace the base values below; other candidates: seed, layer, topology, n_agents


    # model
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    cache_dir: Optional[str] = None  # None -> default HF cache
    dtype: str = "bfloat16"  # name of a torch dtype
    devices: Tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")  # one replica each

    # steering. `layer` = 0-based index into model.model.layers (output of that
    # decoder layer); the paper's "layer 20" (hidden_states idx, row 20 of the file) -> 19 here.
    # Persona vectors: pre-extracted Big Five vectors from Feng et al., ICLR 2026 (HF dataset
    # `persona_dataset`), downloaded by download_model.py. Each file is [n_layers+1, hidden].
    # Poles: inventive/consistent, dependable/careless, outgoing/solitary,
    # compassionate/self-interested, nervous/calm.
    persona_dataset: str = "xiachongfeng/persona"
    persona_traits: Tuple[str, ...] = (  # to download (files are ~0.4MB each)
        "inventive", "consistent", "careless", "outgoing",
        "solitary", "compassionate", "self-interested", "nervous", "calm",
    )
    vector_kind: str = "response_avg_diff"  # or prompt_avg_diff / prompt_last_diff
    persona_trait: str = "careless"  # which trait to steer with (any of persona_traits)
    vectors_dir: str = "vectors"  # relative to the cwd (run from code/)
    layer: int = 19
    # Every steer vector (persona or random) is rescaled to this norm; the steering shift is
    # alpha * steer_vector_norm. 20 ~ raw persona-vector norms at layer 19 (~23 for 'outgoing'),
    # so alpha keeps its old meaning. 1.0 would make alpha the shift magnitude itself.
    steer_vector_norm: float = 20.0
    alpha: float = 1  # steering coefficient
    # 'persona': steer + measure persona vector. 'random': steer + measure a seeded random
    # vector (same norm). 'none': no steering, measure the persona vector (baseline).
    steering_mode: str = "persona"  # 'persona' | 'random' | 'none'

    # swarm
    layer_sizes: Tuple[int, ...] = LAYER_SIZES  # funnel layers; informational (edges/n_agents use LAYER_SIZES)
    n_agents: int = sum(LAYER_SIZES)
    topology: str = "custom"  # 'chain' | 'ring' | 'star' | 'full' | 'custom'
    custom_edges: Tuple[Tuple[int, int], ...] = layered_edges(LAYER_SIZES)  # (src, dst); used if topology == 'custom'
    source_agents: Tuple[int, ...] = (0,)  # steered (unless mode 'none'), seeded, hop 0
    n_turns: int = 6  # number of waves (turn 0 = seed prompt); exchanges happen inside a wave
    # Round trips per edge (S -> R) in a wave: R answers S's latest message, S answers R, R answers...
    # (S makes E-1 replies, R makes E). R talks to its in-neighbors one at a time, by id; its last
    # reply is what its out-neighbors see next wave. 1 = a single reply per in-neighbor.
    exchanges_per_edge: int = 1
    hop_isolated: bool = True  # True: only in-neighbors' latest msgs; False: all unseen msgs
    system_prompts: Tuple[str, ...] = (
        "You're an assistant in a group of agents. Reply in a few sentences to the group to discuss perspectives and coordinate your response.",
    )  # cycled over agent ids
    seed_prompt: str = "Under what conditions could empathy increase political polarization?"

    # generation
    max_new_tokens: int = 2 ** 8 # was 2 ** 7
    temperature: float = 0.2  # 0 -> greedy
    seed: int = 0

    output_dir: str = "results"

    def vector_path(self, trait):
        return f"{self.vectors_dir}/{trait}_{self.vector_kind}.pt"

    @property
    def persona_vector_path(self):
        return self.vector_path(self.persona_trait)


CFG = Config()
