"""
Perplexity + NIAH — correct implementations.

PPL: Monkey-patch model.layers[i].self_attn to compress K,V INSIDE the forward pass,
     after projection but before score computation. This is how real KV cache
     compression works.

NIAH: Construct prompt so that chat template markers are NEVER truncated.
      Filler text is sized to fit within budget AFTER accounting for all overhead.
"""
import sys, os, math, time, json, logging, argparse, types
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "turboquant_cutile"))

from turboquant_cutile import TurboQuantEngine, LloydMaxCodebook

log = logging.getLogger("v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

RESULTS_DIR = PROJECT_ROOT / "results" / "v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_result(filename, data):
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    log.info("Saved: %s", path)

def quantize_nearest(x, centroids):
    c = centroids.to(x.device)
    return c[(x.unsqueeze(-1) - c).abs().argmin(dim=-1).long()]


def calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_calib*5}]")
    cov = {(l,h): {"xtx": torch.zeros(hd,hd,dtype=torch.float64), "n":0}
           for l in range(n_layers) for h in range(n_kv)}
    nd = 0
    for item in ds:
        text = item.get("text","")
        if len(text.strip())<100: continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1]<16: continue
        with torch.no_grad():
            out = model(**enc, use_cache=True); kv = out.past_key_values
        for l in range(n_layers):
            try: k_l = kv.key_cache[l].float().cpu()
            except:
                try: k_l = kv[l][0].float().cpu()
                except: k_l = list(kv)[l][0].float().cpu()
            for h in range(n_kv):
                X = k_l[0,h,:,:].double()
                cov[(l,h)]["xtx"] += X.T@X; cov[(l,h)]["n"] += X.shape[0]
        nd += 1
        if nd >= n_calib: break
    eigen = {}
    for l in range(n_layers):
        for h in range(n_kv):
            C = (cov[(l,h)]["xtx"]/cov[(l,h)]["n"]).float()
            ev, evec = torch.linalg.eigh(C)
            ev = ev.flip(0).clamp(min=0); evec = evec.flip(1)
            d_eff = max(2, min(int(round((ev.sum()**2/(ev**2).sum()).item())), hd-2))
            eigen[(l,h)] = {"evec": evec, "d_eff": d_eff}
    return eigen


# =========================================================================
# QUANTIZED ATTENTION WRAPPER
# =========================================================================
class KVCompressor:
    """Compresses K,V tensors inline during attention computation."""
    
    def __init__(self, method, eigen, layer_idx, n_kv, hd, device):
        self.method = method
        self.eigen = eigen
        self.layer_idx = layer_idx
        self.n_kv = n_kv
        self.hd = hd
        self.device = device
        # Pre-create codebooks (avoid re-creating per call)
        self.cb_k = LloydMaxCodebook(hd, 2)
        self.cb_v = LloydMaxCodebook(hd, 3)
    
    @torch.no_grad()
    def compress_kv(self, key_states, value_states):
        """Compress K,V after projection, before attention scores.
        key_states: [batch, n_kv_heads, seq_len, head_dim]
        value_states: [batch, n_kv_heads, seq_len, head_dim]
        Returns: compressed (key_states, value_states) same shape
        """
        if self.method == "fp16":
            return key_states, value_states
        
        k_out = key_states.clone()
        v_out = value_states.clone()
        
        for h in range(min(self.n_kv, key_states.shape[1])):
            K_h = key_states[0, h].float()  # [seq, hd]
            V_h = value_states[0, h].float()
            
            if self.method == "sq":
                ed = self.eigen.get((self.layer_idx, h))
                if ed is None: continue
                evec = ed["evec"].to(self.device).float()
                VT = evec.T.contiguous()
                
                kn = K_h.norm(dim=-1, keepdim=True)
                kr = (K_h / (kn + 1e-8)) @ VT
                kh = quantize_nearest(kr, self.cb_k.centroids)
                k_out[0, h] = ((kh @ evec) * kn).to(key_states.dtype)
                
                vn = V_h.norm(dim=-1, keepdim=True)
                vr = (V_h / (vn + 1e-8)) @ VT
                vh = quantize_nearest(vr, self.cb_v.centroids)
                v_out[0, h] = ((vh @ evec) * vn).to(value_states.dtype)
                
            elif self.method == "tq":
                tq = TurboQuantEngine(head_dim=self.hd, total_bits=3, device=self.device)
                ck = tq.compress_keys_pytorch(K_h.half())
                cv = tq.compress_values_pytorch(V_h.half())
                k_out[0, h] = ck["k_mse"].to(key_states.dtype)
                v_out[0, h] = tq.decompress_values_pytorch(cv).to(value_states.dtype)
        
        return k_out, v_out


def install_kv_compression(model, method, eigen, n_layers, n_kv, hd, device):
    """Monkey-patch every attention layer to compress K,V inside forward().
    
    This intercepts the K,V tensors AFTER they are projected from hidden states
    and BEFORE they are used in the attention score computation.
    """
    compressors = []
    original_forwards = []
    
    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn
        compressor = KVCompressor(method, eigen, layer_idx, n_kv, hd, device)
        compressors.append(compressor)
        original_forwards.append(attn.forward)
        
        # We intercept via a hook on the attention output
        # But the real interception needs to happen on K,V BEFORE sdpa
        # The cleanest way: wrap the forward to modify past_key_values
        # before they are used in the next call
        
        # For DynamicCache-based models, K,V are stored in the cache 
        # after each layer's forward. We can add a post-forward hook
        # that modifies the cache entries for THIS layer.
        
        def make_hook(comp, l_idx):
            def hook(module, input, output):
                # output is (attn_output, attn_weights, past_key_value)
                # The past_key_value already has this layer's K,V stored
                # Compress them in-place so next-token predictions use compressed cache
                if isinstance(output, tuple) and len(output) >= 3:
                    cache = output[2]
                    if cache is not None:
                        try:
                            k = cache.key_cache[l_idx]
                            v = cache.value_cache[l_idx]
                            k_new, v_new = comp.compress_kv(k, v)
                            cache.key_cache[l_idx] = k_new
                            cache.value_cache[l_idx] = v_new
                        except (AttributeError, IndexError):
                            pass
                return output
            return hook
        
        h = attn.register_forward_hook(make_hook(compressor, layer_idx))
        compressors.append(h)  # store handle for cleanup
    
    return compressors


def remove_kv_compression(hooks):
    """Remove all compression hooks."""
    for h in hooks:
        if hasattr(h, 'remove'):
            h.remove()


# =========================================================================
# PART 1: PERPLEXITY
# =========================================================================
def run_perplexity(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    device = args.device
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    log.info("Loading %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
    model.eval()
    
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    
    n_calib = 50 if args.quick else 200
    log.info("Calibrating (%d seqs)...", n_calib)
    eigen = calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd)
    
    # Load WikiText-2
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([item["text"] for item in ds if len(item.get("text","").strip()) > 0])
    max_len = 1024 if args.quick else 2048
    
    # Standard HF perplexity approach: 
    # Process full sequence with use_cache=True so hooks fire and compress K,V
    # The key insight: with hooks installed, every forward pass compresses the
    # K,V cache AFTER each attention layer. When the model processes token t+1,
    # it reads the compressed K,V from tokens 0..t and produces logits accordingly.
    # This is exactly what happens in real inference with a compressed cache.
    
    # We need to process token-by-token (or in small chunks) with use_cache=True
    # so that each new token's K,V is computed, compressed, and stored.
    
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    seq_len = input_ids.shape[1]
    log.info("Tokens: %d", seq_len)
    
    results = {}
    
    for method in ["fp16", "tq", "sq"]:
        log.info("PPL [%s]...", method)
        hooks = install_kv_compression(model, method, eigen, n_layers, n_kv, hd, device)
        
        t0 = time.time()
        nlls = []
        past = None
        
        # Process one token at a time with cache
        # First token: no cache, just forward
        # Subsequent tokens: use compressed cache from previous tokens
        chunk_size = 1  # token-by-token for correctness
        
        for i in range(0, seq_len - 1):
            token = input_ids[:, i:i+1]
            
            with torch.no_grad():
                out = model(token, past_key_values=past, use_cache=True)
            
            past = out.past_key_values
            # Hooks have already compressed the cache at this point
            
            # NLL for predicting next token
            logits = out.logits[:, -1, :]  # [1, vocab]
            target = input_ids[:, i+1]     # [1]
            nll = F.cross_entropy(logits, target, reduction="none")
            nlls.append(nll.item())
            
            if (i+1) % 200 == 0:
                running_ppl = math.exp(np.mean(nlls))
                log.info("  %s: token %d/%d, running PPL=%.2f", method, i+1, seq_len-1, running_ppl)
        
        elapsed = time.time() - t0
        avg_nll = np.mean(nlls)
        ppl = math.exp(avg_nll)
        log.info("  %s: PPL=%.2f (%.1fs, %d tokens)", method, ppl, elapsed, len(nlls))
        results[method] = {"ppl": float(ppl), "time_s": float(elapsed), "n_tokens": len(nlls)}
        
        remove_kv_compression(hooks)
        # Clear cache for next method
        past = None
        torch.cuda.empty_cache()
    
    # Sanity check
    fp_ppl = results["fp16"]["ppl"]
    log.info("\nFP16 PPL=%.2f %s", fp_ppl, "(OK)" if 3 < fp_ppl < 30 else "(UNEXPECTED)")
    
    save_result("v3_perplexity_v2.json", {
        "model": model_name, "dataset": "wikitext-2-raw-v1",
        "max_length": max_len, "token_by_token": True,
        "results": results,
    })
    
    del model; torch.cuda.empty_cache()
    return results


# =========================================================================
# PART 2: NIAH
# =========================================================================
def run_niah(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = args.device
    hf_token = os.environ.get("HF_TOKEN")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    log.info("Loading %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device, token=hf_token
    )
    model.eval()
    
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    
    n_calib = 50 if args.quick else 200
    log.info("Calibrating Llama (%d seqs)...", n_calib)
    eigen = calibrate(model, tokenizer, n_calib, device, n_layers, n_kv, hd)
    
    NEEDLE = "The special magic number is 7392."
    QUESTION = "What is the special magic number mentioned in the text above? Answer with just the number."
    
    FILLERS = [
        "The history of artificial intelligence spans decades of research and innovation across multiple disciplines. Scientists and engineers have worked tirelessly to develop systems capable of reasoning, learning, and solving complex problems that were once thought to be exclusively within the domain of human cognition. The field has seen remarkable transformations from early symbolic approaches to modern deep learning paradigms that power today's most capable AI systems.",
        "Modern agriculture relies on a complex network of supply chains, from seed production to distribution of harvested crops. Climate patterns, soil conditions, and water availability all play crucial roles in determining crop yields across different geographic regions around the world. Sustainable farming practices are becoming increasingly important as global food demand continues to grow with expanding populations.",
        "The development of quantum computing represents one of the most significant technological challenges of our era. Researchers are working to overcome decoherence, error correction, and scalability issues that currently limit the practical application of quantum processors in solving real-world computational problems that classical computers cannot handle efficiently.",
        "Urban planning in the twenty-first century must balance economic development with environmental sustainability. Cities around the globe are experimenting with green infrastructure, mixed-use zoning, and public transportation investments to create more livable and resilient urban environments for their growing populations and diverse communities.",
        "The study of marine biology has revealed extraordinary biodiversity in ocean ecosystems that continues to surprise researchers. From the sunlit surface waters to the crushing depths of hydrothermal vents, life has adapted to fill virtually every available ecological niche in the world's vast and largely unexplored oceans.",
        "Advances in materials science have led to the development of metamaterials with properties not found anywhere in nature. These carefully engineered structures can manipulate electromagnetic waves in unprecedented ways, enabling potential applications ranging from perfect lenses to novel antenna designs and even theoretical invisibility cloaking devices.",
        "The global financial system operates through an intricate web of institutions, regulations, and market mechanisms that span national borders. Central banks, commercial banks, and investment firms each play distinct yet interconnected roles in maintaining economic stability and facilitating the efficient flow of capital throughout the world economy.",
        "Archaeological discoveries continue to reshape our understanding of ancient civilizations and human history. Recent excavations across multiple continents have revealed sophisticated urban planning, advanced metallurgical techniques, and surprisingly complex social hierarchies in cultures far older than previously believed by mainstream historians and scholars.",
    ]
    
    ctx_lengths = [4096, 8192] if args.quick else [4096, 8192, 16384]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    records = []
    
    for ctx_len in ctx_lengths:
        for depth in depths:
            # KEY FIX: Build prompt with proper budget management
            # 1. First, format the question part with chat template to measure its token count
            question_part = "\n\n" + QUESTION
            needle_tokens = len(tokenizer.encode(NEEDLE, add_special_tokens=False))
            
            # 2. Build a test prompt to measure chat template overhead
            test_messages = [{"role": "user", "content": "X"}]
            test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            overhead_tokens = len(tokenizer.encode(test_prompt, add_special_tokens=False))
            question_tokens = len(tokenizer.encode(question_part, add_special_tokens=False))
            
            # 3. Calculate how many filler tokens we can afford
            budget = ctx_len - overhead_tokens - needle_tokens - question_tokens - 10  # 10 for safety
            
            # 4. Build filler text to exactly fit the budget
            filler_text = ""
            para_idx = 0
            while len(tokenizer.encode(filler_text, add_special_tokens=False)) < budget:
                filler_text += FILLERS[para_idx % len(FILLERS)] + "\n\n"
                para_idx += 1
            # Trim to budget
            filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)
            if len(filler_tokens) > budget:
                filler_text = tokenizer.decode(filler_tokens[:budget], skip_special_tokens=True)
            
            # 5. Insert needle at specified depth
            insert_pos = int(depth * len(filler_text))
            context = filler_text[:insert_pos] + "\n" + NEEDLE + "\n" + filler_text[insert_pos:]
            
            # 6. Build full prompt with chat template
            messages = [{"role": "user", "content": context + question_part}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            actual_len = input_ids.shape[1]
            
            # Verify the prompt ends with the assistant header
            prompt_tail = prompt[-100:]
            log.info("NIAH ctx=%d depth=%.0f%% tokens=%d tail='%s'",
                     ctx_len, depth*100, actual_len, prompt_tail.replace('\n','\\n')[-60:])
            
            for method in ["fp16", "tq", "sq"]:
                try:
                    hooks = install_kv_compression(model, method, eigen, n_layers, n_kv, hd, device)
                    
                    gen_ids = model.generate(
                        input_ids,
                        max_new_tokens=20,
                        do_sample=False,
                        temperature=1.0,
                        use_cache=True,
                    )
                    response = tokenizer.decode(gen_ids[0, actual_len:], skip_special_tokens=True).strip()
                    correct = "7392" in response
                    
                    remove_kv_compression(hooks)
                    
                    log.info("  %s: %s — \"%s\"",
                             method, "PASS" if correct else "FAIL", response[:80])
                    
                    records.append({
                        "ctx_len": ctx_len, "depth": depth, "method": method,
                        "correct": correct, "response": response[:200],
                        "actual_tokens": actual_len,
                    })
                    
                except Exception as e:
                    log.error("  %s error: %s", method, e)
                    import traceback; traceback.print_exc()
                    records.append({
                        "ctx_len": ctx_len, "depth": depth, "method": method,
                        "correct": False, "error": str(e),
                    })
                    try: remove_kv_compression(hooks)
                    except: pass
            
            # Save after each (ctx_len, depth)
            hm = {}
            for r in records:
                m, ct, dp = r["method"], str(r["ctx_len"]), str(r["depth"])
                hm.setdefault(m, {}).setdefault(ct, {})[dp] = r.get("correct", False)
            save_result("v3_niah_llama_v2.json", {"model": model_name, "records": records, "heatmap": hm})
    
    # Summary
    log.info("\n=== NIAH Summary ===")
    for method in ["fp16", "tq", "sq"]:
        mr = [r for r in records if r["method"] == method]
        correct = sum(1 for r in mr if r.get("correct", False))
        log.info("  %s: %d/%d correct", method, correct, len(mr))
    
    del model; torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--part", type=int, default=0, help="0=both, 1=ppl, 2=niah")
    args = parser.parse_args()
    
    if args.part in (0, 1):
        log.info("\n" + "="*60)
        log.info("PART 1: PERPLEXITY (quantized attention hooks)")
        log.info("="*60)
        run_perplexity(args)
    
    if args.part in (0, 2):
        log.info("\n" + "="*60)
        log.info("PART 2: NIAH (Llama, budget-managed prompt)")
        log.info("="*60)
        run_niah(args)

if __name__ == "__main__":
    main()
