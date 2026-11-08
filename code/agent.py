"""One Qwen replica on one device, with steering + activation-measurement hooks.

LAYER CONVENTION (the only one used in this codebase): `layer` is the 0-based
index into model.model.layers; hooks act on the OUTPUT of that decoder layer.
Persona-vector files (HF dataset xiachongfeng/persona) use the hidden_states
convention (row 0 = embeddings; checked: row-0 norm ~0.04), so decoder layer l ==
row l + 1 of those files. `load_persona_vector` does the +1. Avoid the last layer:
HF's final hidden_states entry has the final norm applied, so it isn't layers[-1]'s output.

Vector file format: torch tensor, [n_layers + 1, hidden] (downloaded format) or [hidden]
(already for the chosen layer).
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


@dataclass
class Replica:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str


def load_replica(cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir)
    # No device_map / low_cpu_mem_usage: both require accelerate. Load fully on CPU
    # (~15 GB RAM transiently per replica), then .to(device). Call sequentially.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        cache_dir=cfg.cache_dir,
        torch_dtype=getattr(torch, cfg.dtype),  # `dtype=` kwarg only exists in new transformers
    ).to(device).eval()
    return Replica(model, tokenizer, device)


def scale_to_norm(vector, norm):
    """THE place vectors get their magnitude: steer vectors all have norm cfg.steer_vector_norm,
    so alpha * vector has the same size whatever the trait (or random direction)."""
    return vector.float() * (norm / vector.float().norm())


def load_persona_vector(path, layer, norm):
    """Returns float32 [hidden] vector for decoder layer `layer`, rescaled to `norm`."""
    vector = torch.load(path, map_location="cpu").float()
    return scale_to_norm(vector[layer + 1] if vector.ndim == 2 else vector, norm)


def _encode(tokenizer, messages, add_generation_prompt):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def _hidden(output):
    # decoder layers return a tuple in older transformers, a tensor in newer ones
    return output[0] if isinstance(output, tuple) else output


def _steer_hook(vector, alpha):
    def hook(module, args, output):
        shifted = _hidden(output) + alpha * vector.to(_hidden(output))  # all positions
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    return hook


def _record_hook(records):
    def hook(module, args, output):  # no return -> output untouched
        records.append(_hidden(output)[0, -1].detach().float().cpu())

    return hook


def _summarize(records, vector):
    # records[0]: prefill call -> last prompt token. records[1:]: one per decode step.
    acts = torch.stack(records)
    proj = acts @ (vector.float() / vector.norm())  # scalar projection on unit vector
    cos = proj / acts.norm(dim=-1)
    gen_proj, gen_cos = (proj[1:].mean().item(), cos[1:].mean().item()) if len(proj) > 1 else (math.nan,) * 2
    return {"prompt_proj": proj[0].item(), "prompt_cos": cos[0].item(),
            "gen_proj": gen_proj, "gen_cos": gen_cos}


def derive_seed(base_seed, agent_id, turn):
    """Deterministic, well-mixed sampling seed per (base seed, agent, turn)."""
    return int(np.random.SeedSequence([base_seed, agent_id, turn]).generate_state(1)[0])


def _seeded_sampler(seed, device, temperature):
    """Logits processor that draws the next token from its own device generator and masks all
    other tokens, so HF's internal (global-RNG) multinomial is forced to pick that token.
    transformers' generate() has no `generator=` arg, and the global RNG is shared by threads."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def sample(input_ids, scores):
        probs = torch.softmax(scores.float() / temperature, dim=-1)
        token = torch.multinomial(probs, 1, generator=generator)
        return torch.full_like(scores, -torch.inf).scatter(1, token, 0.0)

    return sample


@torch.no_grad()
def generate_reply(replica, messages, cfg, steer_vector=None, measure_vector=None, *, agent_id, turn):
    """Reply to chat `messages` (system + history, last = user). Returns (text, measurement).

    Sampling is seeded from derive_seed(cfg.seed, agent_id, turn): distinct per agent and turn,
    reproducible per config, independent of thread scheduling.

    steer_vector: add cfg.alpha * vector at cfg.layer (None -> unsteered).
    measure_vector: project cfg.layer activations onto it (None -> empty measurement).
    The measure hook is registered AFTER the steer hook, so it sees the steered
    stream; for unsteered agents that is just the raw stream.
    """
    layer_module = replica.model.model.layers[cfg.layer]
    records, handles = [], []
    try:
        if steer_vector is not None:
            handles.append(layer_module.register_forward_hook(_steer_hook(steer_vector, cfg.alpha)))
        if measure_vector is not None:
            handles.append(layer_module.register_forward_hook(_record_hook(records)))
        input_ids = _encode(replica.tokenizer, messages, True).to(replica.device)
        sampling = cfg.temperature > 0
        output_ids = replica.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=cfg.max_new_tokens,
            do_sample=sampling,  # temperature 0 -> greedy: deterministic, seeds irrelevant
            temperature=None,  # applied inside the sampler processor instead
            top_p=None, top_k=None,
            logits_processor=LogitsProcessorList([_seeded_sampler(
                derive_seed(cfg.seed, agent_id, turn), replica.device, cfg.temperature)]) if sampling else None,
            pad_token_id=replica.tokenizer.eos_token_id,
        )
    finally:
        for handle in handles:
            handle.remove()
    text = replica.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    return text, (_summarize(records, measure_vector) if records else {})


@torch.no_grad()
def extract_persona_vector(replica, trait_chats, neutral_chats, cfg):
    """Diff-in-means at cfg.layer over response tokens. Each chat = messages ending
    with an assistant reply. Returns a 1-D [hidden] vector with norm cfg.steer_vector_norm
    (save with torch.save; loading rescales again, which is a no-op)."""
    def mean_activation(chats):
        means = []
        for chat in chats:
            n_prompt = _encode(replica.tokenizer, chat[:-1], True).shape[1]
            ids = _encode(replica.tokenizer, chat, False).to(replica.device)
            hidden = replica.model(ids, output_hidden_states=True).hidden_states[cfg.layer + 1]
            means.append(hidden[0, n_prompt:].float().mean(0).cpu())
        return torch.stack(means).mean(0)

    diff = mean_activation(trait_chats) - mean_activation(neutral_chats)
    return scale_to_norm(diff, cfg.steer_vector_norm)
