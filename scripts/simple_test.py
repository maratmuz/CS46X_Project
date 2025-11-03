from evo2 import Evo2

# evo2_model = Evo2('evo2_7b')
evo2_model = Evo2('evo2_40b_base')

output = evo2_model.generate(prompt_seqs=["ACGT"], n_tokens=400, temperature=1.0, top_k=4)

print(output.sequences[0])