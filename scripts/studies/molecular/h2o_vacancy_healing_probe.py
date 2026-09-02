"""H2O vacancy-healing probe on TiO2(110): does water's own O fill a bridging-oxygen
vacancy while a proton transfers to a *different*, still-intact neighboring O2c?

Textbook mechanism (Wendt et al., Science 2005): a bridging-O vacancy leaves excess
electron density partly localized on nearby Ti, making the neighboring intact O2c a
better proton acceptor than on the pristine surface (h2o_dissociation_probe.py's
clean-surface case) -- water heals the vacancy with its own O, then dissociates onto
that activated neighbor. Two different questions, two different starting guesses:

- ``"direct_fill"``: place water's O exactly where the removed O2c used to sit --
  the most literal "heal the gap" guess.
- ``"bridge_site"``: place it at whatever bridge site pymatgen's own site-finder
  identifies nearest the vacancy on the defective slab, instead of assuming the old
  atom's exact former position is still the right target after the lattice relaxes
  around the defect.

Both then get oriented via h2o_dissociation_probe.orient_toward(mode="bisector") toward
the nearest *remaining* O2c, relaxed, then nudged and re-relaxed -- reusing that script's
orient_toward()/is_dissociated()/nudge_apart() directly since none of them assume
anything about how the water got placed, only about the indices once it's there.

    python scripts/studies/molecular/h2o_vacancy_healing_probe.py runs/h2o_vacancy_healing_probe
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from oxide_workflow import pipeline
from oxide_workflow.backends import get_backend, relax
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.energetics import adsorption_energy, gas_reference_energy
from oxide_workflow.stages import (
    _site_environments,
    adsorbate_candidates,
    apply_bottom_freeze,
    exposed_surface_atoms,
    make_slab,
    oxygen_vacancy_candidates,
    site_label,
)
from oxide_workflow.structures import get_structure

from h2o_dissociation_probe import is_dissociated, min_image_vector, nudge_apart, orient_toward

MP_ID = "mp-2657"
MODELS = ["MACE-mh1-omat", "UMA-oc22", "SevenNet-omni-omat24", "UMA-M-omat"]
FMAX = 0.02
PLACEMENT_MODES = ("direct_fill", "bridge_site")
DESORB_CHECK_STEP = 100
DESORB_TREND_WINDOW = 20
EXTEND_STEPS = 100
MAX_EXTENSIONS = 3


def _min_image_distance(lattice, cart_a: np.ndarray, cart_b: np.ndarray) -> float:
    return float(np.linalg.norm(min_image_vector(lattice, cart_a, cart_b)))


def bridging_o2c_vacancy(pristine: Structure, freeze_bottom_fraction: float):
    """The single O2c vacancy candidate -- oxygen_vacancy_candidates() returns one
    VacancyCandidate per symmetry-distinct removable O; this is the same site type
    Track 1's OVFE benchmark targets (site_label == "O2c"), just reused here for a
    different purpose (healing) instead of a formation-energy comparison."""
    candidates = oxygen_vacancy_candidates(
        pristine, freeze_bottom_fraction=freeze_bottom_fraction,
    )
    o2c_candidates = [c for c in candidates if c.site_id["site_label"] == "O2c"]
    if not o2c_candidates:
        raise RuntimeError("no O2c (bridging) vacancy candidate found")
    return o2c_candidates[0]


def find_remaining_o2c(vacancy_structure: Structure, gap_cart: np.ndarray) -> int:
    """Nearest still-intact O2c to the vacancy gap, searched on the *defective* slab --
    the removed atom is simply absent from this structure's indexing, so it can never be
    picked by accident the way it could if this searched the pristine structure."""
    z = vacancy_structure.cart_coords[:, 2]
    exposed = exposed_surface_atoms(vacancy_structure, depth=float(z.max() - z.min()))
    environments = _site_environments(vacancy_structure)
    o2c_candidates = [
        i for i in exposed
        if environments[i][0] == "O" and site_label(*environments[i][::2]) == "O2c"
    ]
    if not o2c_candidates:
        raise RuntimeError("no remaining O2c found near the vacancy")
    lattice = vacancy_structure.lattice
    return min(
        o2c_candidates,
        key=lambda i: _min_image_distance(lattice, vacancy_structure.cart_coords[i], gap_cart),
    )


def place_water_at(vacancy_structure: Structure, gap_cart: np.ndarray, freeze_bottom_fraction: float):
    """Append H2O (default template, O at ``gap_cart``) directly -- no AdsorbateSiteFinder
    involved, since we already know exactly where it should go (either the old vacancy
    position or a caller-supplied bridge-site coordinate). Mirrors adsorbate_candidates'
    own "adsorbate atoms appended last, then bottom-frozen" convention so downstream code
    (anchor_idx = len(structure) - n_ads, orient_toward, etc.) works unmodified."""
    species, coords = ADSORBATE_FRAGMENTS["H2O"]
    new_species = list(vacancy_structure.species) + list(species)
    new_coords = list(vacancy_structure.cart_coords) + [gap_cart + np.array(c) for c in coords]
    placed = Structure(vacancy_structure.lattice, new_species, new_coords, coords_are_cartesian=True)
    n_ads = len(species)
    apply_bottom_freeze(
        placed, freeze_bottom_fraction, always_free=set(range(len(placed) - n_ads, len(placed))),
    )
    return placed


def bridge_site_near_gap(vacancy_structure: Structure, gap_cart: np.ndarray, cfg: RunConfig):
    """Cartesian coordinate of whichever pymatgen-found bridge site sits nearest the
    vacancy gap -- an alternative to assuming the old atom's exact former position is
    still the right target once the lattice has had a chance to relax around the defect.
    Reuses adsorbate_candidates' own bridge-site enumeration (positions=("bridge",)); a
    throwaway "H" adsorbate config is enough since only the site coordinates are used,
    not the placed fragment itself (place_water_at does that separately)."""
    h_species, h_coords = ADSORBATE_FRAGMENTS["H"]
    candidates = adsorbate_candidates(
        vacancy_structure,
        replace(cfg.adsorbate, species=h_species, coords=h_coords, positions=("bridge",)),
        freeze_bottom_fraction=cfg.slab.freeze_bottom_fraction,
    )
    lattice = vacancy_structure.lattice
    nearest = min(
        candidates,
        key=lambda c: _min_image_distance(lattice, lattice.get_cartesian_coords(c.site_id["frac_coord"]), gap_cart),
    )
    return lattice.get_cartesian_coords(nearest.site_id["frac_coord"])


def run_one(model: str, outdir: Path, cfg: RunConfig) -> list[dict]:
    backend = get_backend(model)
    odir = outdir / model

    def relax_record(structure, stage, source_desc, relax_cell=False):
        out = pipeline._relax_record(
            structure, backend, stage=stage, protocol="reference",
            geometry_source=source_desc, cfg=cfg, outdir=odir,
            relax_cell=relax_cell, canonical=True,
        )
        print(f"    {stage:14s} E={out.energy:.4f} eV  {out.header['nsteps']} steps  "
              f"{out.elapsed_s:.0f}s", flush=True)
        return out

    print(f"[{model}]", flush=True)
    start = get_structure(MP_ID)
    bulk = relax_record(start, "bulk", "db", relax_cell=True).structure
    slab_in = make_slab(bulk, cfg.slab)
    slab_out = relax_record(slab_in, "slab", "cut_from_relaxed_bulk")
    pristine_energy = slab_out.energy
    pristine_structure = slab_out.structure

    vac_cand = bridging_o2c_vacancy(pristine_structure, cfg.slab.freeze_bottom_fraction)
    gap_cart = pristine_structure.lattice.get_cartesian_coords(vac_cand.site_id["frac_coord"])
    vacancy_out = relax_record(vac_cand.structure, "vacancy", "o2c_removed_from_relaxed_slab")
    vacancy_energy = vacancy_out.energy
    vacancy_structure = vacancy_out.structure
    print(f"    vacancy site: {vac_cand.site_id['site_label']} at {gap_cart}", flush=True)

    h2o_species, h2o_coords = ADSORBATE_FRAGMENTS["H2O"]
    n_ads = len(h2o_species)
    e_gas = gas_reference_energy(
        backend, cfg, pipeline.relax, species=h2o_species, coords=h2o_coords,
    )

    rows = []
    for placement in PLACEMENT_MODES:
        target_cart = (
            gap_cart if placement == "direct_fill"
            else bridge_site_near_gap(vacancy_structure, gap_cart, cfg)
        )
        placed = place_water_at(vacancy_structure, target_cart, cfg.slab.freeze_bottom_fraction)
        anchor_idx = len(placed) - n_ads
        h_near_idx = anchor_idx + 1

        remaining_o2c_idx = find_remaining_o2c(vacancy_structure, target_cart)
        target_dir = min_image_vector(
            vacancy_structure.lattice, vacancy_structure.cart_coords[remaining_o2c_idx], target_cart,
        )
        target_dir = target_dir / np.linalg.norm(target_dir)
        oriented = orient_toward(placed, anchor_idx, n_ads, target_dir, mode="bisector")

        res = relax(
            oriented, backend, workdir=odir / "adsorbate" / placement,
            fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
            desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
            desorb_trend_window=DESORB_TREND_WINDOW,
            extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
        )
        e_ads = adsorption_energy(res.energy, vacancy_energy, e_gas)
        dissociated = is_dissociated(res.structure, h_near_idx, anchor_idx, remaining_o2c_idx)
        print(f"    [{placement}] E_ads={e_ads:+.4f} eV  dissociated={dissociated}  "
              f"converged={res.converged}  nsteps={res.nsteps}", flush=True)

        nudged = nudge_apart(res.structure, h_near_idx, remaining_o2c_idx)
        res_nudged = relax(
            nudged, backend, workdir=odir / "adsorbate" / placement / "nudged",
            fmax=cfg.relax.fmax, max_steps=cfg.relax.max_steps, optimizer=cfg.relax.optimizer,
            desorb_check_n_ads=n_ads, desorb_check_step=DESORB_CHECK_STEP,
            desorb_trend_window=DESORB_TREND_WINDOW,
            extend_if_approaching=True, extend_steps=EXTEND_STEPS, max_extensions=MAX_EXTENSIONS,
        )
        e_ads_nudged = adsorption_energy(res_nudged.energy, vacancy_energy, e_gas)
        dissociated_nudged = is_dissociated(res_nudged.structure, h_near_idx, anchor_idx, remaining_o2c_idx)
        print(f"    [{placement}] nudged E_ads={e_ads_nudged:+.4f} eV  "
              f"dissociated={dissociated_nudged}  converged={res_nudged.converged}  "
              f"nsteps={res_nudged.nsteps}  dE_from_nudge={e_ads_nudged - e_ads:+.4f} eV", flush=True)

        rows.append({
            "model": model, "placement": placement,
            "vacancy_site_label": vac_cand.site_id["site_label"],
            "remaining_o2c_idx": remaining_o2c_idx,
            "e_ads_eV": e_ads, "dissociated": dissociated,
            "converged": res.converged, "nsteps": res.nsteps,
            "e_ads_nudged_eV": e_ads_nudged, "dissociated_nudged": dissociated_nudged,
            "converged_nudged": res_nudged.converged, "nsteps_nudged": res_nudged.nsteps,
            "d_e_nudge_eV": e_ads_nudged - e_ads,
        })

    return rows


def main(outdir: Path, models: list[str]) -> None:
    cfg = RunConfig()
    cfg = replace(cfg, relax=replace(cfg.relax, fmax=FMAX))
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.jsonl"
    for model in models:
        rows = run_one(model, outdir, cfg)
        with results_path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    a = ap.parse_args()
    main(a.outdir, a.models)
