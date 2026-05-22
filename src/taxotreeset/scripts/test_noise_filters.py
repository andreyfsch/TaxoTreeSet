"""Sanity check: aplica NoiseFilter a uma lista conhecida de táxons
problemáticos do census e verifica que casam como esperado."""
import sys
sys.path.insert(0, "src")

from taxotreeset.io.noise_filter import NoiseFilter

nf = NoiseFilter("configs/noise_patterns.json")

# (name, rank, expected_to_be_noise)
test_cases = [
    # Devem casar
    ("unclassified Caudoviricetes", "no rank", True),
    ("unclassified Begomovirus", "no rank", True),
    ("unclassified bacterial viruses", "no rank", True),
    ("Caudoviricetes incertae sedis", "no rank", True),
    ("Viruses incertae sedis", "no rank", True),
    ("Autographivirales incertae sedis", "no rank", True),
    ("environmental samples", "no rank", True),
    ("unclassified RNA viruses ShiM-2016", "no rank", True),
    ("Norovirus GII.4 isolates", "no rank", True),
    ("Norovirus genogroup 1 isolates", "no rank", True),
    ("Influenza A virus with incomplete names", "no rank", True),
    ("H1N1 subtype", "serotype", True),  # via rank blacklist
    ("HIV-1 M:CRF", "no rank", True),

    # NÃO devem casar (táxons biológicos legítimos)
    ("Caudoviricetes", "class", False),
    ("Begomovirus", "genus", False),
    ("Orthornavirae", "kingdom", False),
    ("Picornaviridae", "family", False),
    ("Escherichia coli", "species", False),

    # Casos limítrofes a checar
    ("Unclassified", "genus", True),    # case-insensitive prefix
    ("Some unclassified thing in middle", "genus", False),  # ^ exigido
]

ok = 0
fail = 0
for name, rank, expected in test_cases:
    actual = nf.is_noise(name, rank)
    mark = "✓" if actual == expected else "✗"
    if actual == expected:
        ok += 1
    else:
        fail += 1
    explanation = nf.explain(name, rank) if actual else ""
    print(f"  {mark}  is_noise({name!r:50}, {rank!r:10}) → {actual:5} (esperado {expected})  {explanation}")

print(f"\n{ok}/{ok+fail} OK, {fail} falhas")
print(f"\nEstatísticas: {nf.stats()}")
sys.exit(0 if fail == 0 else 1)