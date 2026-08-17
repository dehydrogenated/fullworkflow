"""Direct test: does each model's own O2 prediction match the real O2 molecule?

Computes each model's O2 dissociation energy, D_e = 2*E(O_atom) - E(O2), and compares it
to the experimental value (5.26 eV, ZPE-removed, NIST -- the same number Kowalski, Meyer &
Marx cite and correct for in PRB 79, 115410 (2009), Sec. II). Unlike substituting a foreign
DFT paper's absolute mu_O into E_vac (which mixes two unrelated energy scales -- see
energetics.py's docstrings on why only same-calculator terms cancel), this stays entirely
within one model's own energy scale: E(O2) and E(O_atom) both come from that model, so their
difference is a portable, code-independent number directly comparable to experiment.

Caveat carried over from energetics.py: ground-state atomic O is an open-shell triplet
(3P), and none of these models carry a spin channel, so the atomic term is the least
trustworthy part of this test -- but it is the direct way to check the hypothesis "the
models are bad at predicting gas-phase O2," rather than inferring it indirectly.

    python scripts/ovfe_o2_dissociation_check.py
"""

from __future__ import annotations

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import gas_reference_energy

MODELS = ["MACE-mh1-omat", "MACE-mh1-oc20", "UMA-omat", "UMA-oc22", "CHGNet-0.3.0", "Orb-v2"]
EXPERIMENTAL_DE_O2 = 5.26  # eV, ZPE-removed, NIST (Kowalski et al. Ref. 61)


def main() -> None:
    cfg = RunConfig()
    print(f"experimental D_e(O2) = {EXPERIMENTAL_DE_O2} eV (ZPE-removed, NIST)\n")
    print(f"{'model':16s}{'E(O2)':>10s}{'E(O atom)':>12s}{'D_e model':>12s}{'error':>10s}")
    for model in MODELS:
        backend = get_backend(model)
        o2_species, o2_coords = ADSORBATE_FRAGMENTS["O2"]
        o_species, o_coords = ADSORBATE_FRAGMENTS["O"]
        e_o2 = gas_reference_energy(backend, cfg, pipeline.relax, species=o2_species, coords=o2_coords)
        e_o = gas_reference_energy(backend, cfg, pipeline.relax, species=o_species, coords=o_coords)
        de = 2 * e_o - e_o2
        error = de - EXPERIMENTAL_DE_O2
        print(f"{model:16s}{e_o2:>10.3f}{e_o:>12.3f}{de:>12.3f}{error:>+10.3f}")


if __name__ == "__main__":
    main()
