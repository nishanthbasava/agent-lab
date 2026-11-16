import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

print(model)  # 12 blocks, each with attn + mlp
for name, p in model.named_parameters():
    print(name, tuple(p.shape))

ids = tok("The Eiffel Tower is in the city of", return_tensors="pt").input_ids
with torch.no_grad():
    out = model(ids, output_hidden_states=True)

hs = out.hidden_states      # tuple of 13: embeddings + output of each block
print(hs[6].shape)          # [1, seq_len, 768]: residual stream after block 6
print(hs[-1])

# 3. Logit lens: decode each layer’s residual stream as if it were the final one.

for i, h in enumerate(hs):
    logits = model.lm_head(model.transformer.ln_f(h[0, -1]))
    print(i, repr(tok.decode(logits.argmax())))

# You’ll see the prediction go from junk to “ Paris” as the layers progress.

# 4. Intervene with a hook: zero out a block’s MLP and see what breaks.
def zero_mlp(module, inp, out):
    return torch.zeros_like(out)

handle = model.transformer.h[8].mlp.register_forward_hook(zero_mlp)
with torch.no_grad():
    print(tok.decode(model(ids).logits[0, -1].argmax()))
handle.remove()

'''
Loop that over all 12 layers and measure how much the correct token’s probability drops. That’s a basic ablation experiment.

Once steps 2 and 4 feel routine, you have what each technique needs:

Probe: run step 2 over a labeled dataset, save hs[layer], and fit logistic regression.
SAE: run step 2 over a lot of text and train the autoencoder on the saved vectors.
LoRA: swap model.transformer.h[i].attn.c_attn for your wrapped version and train.

One terminology note: what hidden_states returns is the residual stream at each layer. Papers will also refer to attention outputs and MLP outputs, which are the pieces added into the residual stream. Those need hooks like the one in step 4 to capture.

TransformerLens wraps all of this more conveniently (run_with_cache, named hook points), but do it with raw hooks once first so you know what the library is doing.
'''