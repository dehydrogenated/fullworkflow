"""Adsorption energetics — turning raw total energies into E_ads.

    E_ads = E(substrate + adsorbate) - E(bare substrate) - E(isolated fragment)

with all three terms from the *same* calculator. That constraint is the whole point.
MLIP total energies carry a large per-atom reference offset — two heads of the same
checkpoint disagree by tens of eV on an identical cell — so raw totals are not comparable
across models. The three terms above balance atom for atom, so the offset cancels exactly
in the difference and E_ads *is* comparable. It is also the number the screening question
actually asks: how strongly does this fragment bind, and at which site.

What does **not** cancel is each model's own error on the isolated gas-phase fragment. A
molecule in vacuum is far from the condensed-matter data these models are trained on, and
an isolated *atom* (H, O) is further still — no MLIP here carries spin state, so an atomic
reference is the shakiest term in the expression. Two consequences worth keeping straight:

- Within one model the gas term is a **constant shift**, identical for every site, facet
  and material. Site rankings, binding-site selection and E_ads *differences* are
  therefore unaffected by how wrong it is.
- Between models it does **not** cancel, so an absolute E_ads gap of a few tenths of an eV
  between two models may be gas-reference error rather than surface chemistry.

Swapping in a DFT gas reference later fixes the second point without touching anything
else: it is one number per (model, fragment), and ``gas_reference_energy`` is the single
place it enters. The on-disk cache below is already keyed that way.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from pymatgen.core import Lattice, Structure

# Å; cubic cell edge for the isolated fragment. Large enough that a fragment at the centre
# does not see its periodic images (the biggest fragment here spans ~1.2 Å, leaving >13 Å
# of vacuum), while staying small enough that the reference relaxation is seconds, not
# minutes.
GAS_BOX = 15.0

# Gas references depend only on (model, fragment, fmax) — never on the material, facet or
# site — so one shared cache serves every run and every member of a sweep.
GAS_CACHE_DIR = Path("runs/_gas_refs")


def fragment_key(species: tuple[str, ...], coords: tuple[tuple[float, float, float], ...]) -> str:
    """Stable cache token for a fragment: its formula plus a digest of its geometry.

    The geometry has to be in the key, not just the formula: two O atoms 1.21 Å apart
    (O2) and the same two atoms far apart are different reference states with very
    different energies, and both would otherwise hash to "OO".
    """
    geom = ";".join(f"{x:.4f},{y:.4f},{z:.4f}" for x, y, z in coords)
    return f"{''.join(species)}_{hashlib.sha1(geom.encode()).hexdigest()[:8]}"


def gas_structure(
    species: tuple[str, ...],
    coords: tuple[tuple[float, float, float], ...],
    box: float = GAS_BOX,
) -> Structure:
    """The fragment alone, centred in a large cubic box — the isolated reference state."""
    centre = np.array([box / 2.0] * 3)
    return Structure(
        Lattice.cubic(box),
        list(species),
        [np.asarray(c, dtype=float) + centre for c in coords],
        coords_are_cartesian=True,
    )


def gas_reference_energy(
    backend,
    cfg,
    relax_fn: Callable,
    cachedir: str | Path = GAS_CACHE_DIR,
    species: tuple[str, ...] | None = None,
    coords: tuple[tuple[float, float, float], ...] | None = None,
) -> float:
    """Relaxed total energy of an isolated fragment for this model.

    Defaults to the configured adsorbate; pass ``species``/``coords`` for a different
    fragment (the oxygen chemical potential needs O2 even when the adsorbate is CO).

    Cheap (a handful of atoms) but not free, and identical for every site, protocol,
    facet and material in a sweep — so it is computed once and cached on disk.

    ``fmax`` is part of the cache key deliberately. E_ads and E_vac each mix three
    energies, so a gas reference converged to a looser tolerance than the run that uses it
    leaks that difference straight into the result; silently reusing a loose reference for
    a tight run would be a real error.

    ``relax_fn`` is injected rather than imported so the caller controls the relaxation
    seam (the pipeline passes its own, which is what the test suite fakes).
    """
    if species is None or coords is None:
        species, coords = cfg.adsorbate.species, cfg.adsorbate.coords
    key = fragment_key(species, coords)
    cachedir = Path(cachedir)
    cache = cachedir / f"gas_{backend.name}_{key}_fmax{cfg.relax.fmax}.json"
    if cache.exists():
        return float(json.loads(cache.read_text())["energy"])

    res = relax_fn(
        gas_structure(species, coords),
        backend,
        relax_cell=False,  # the box is a container, not a physical cell — never relax it
        fmax=cfg.relax.fmax,
        max_steps=cfg.relax.max_steps,
        optimizer=cfg.relax.optimizer,
    )
    cachedir.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "model": backend.name,
                "fragment": key,
                "species": list(species),
                "coords": [list(c) for c in coords],
                "box": GAS_BOX,
                "fmax": cfg.relax.fmax,
                "optimizer": cfg.relax.optimizer,
                "energy": res.energy,
                "nsteps": res.nsteps,
                "converged": res.converged,
            },
            indent=2,
        )
    )
    return float(res.energy)


def adsorption_energy(
    e_total: Optional[float], e_substrate: Optional[float], e_gas: Optional[float]
) -> Optional[float]:
    """E_ads = E(substrate+adsorbate) - E(bare substrate) - E(isolated fragment), in eV.

    Negative means bound. ``None`` if any term is missing, so a stage without a reference
    state simply reports no E_ads rather than a wrong one.
    """
    if e_total is None or e_substrate is None or e_gas is None:
        return None
    return float(e_total - e_substrate - e_gas)


# --- Oxygen vacancy formation ------------------------------------------------------------

# Which reservoir the removed oxygen atom is returned to. This is a *convention*, not a
# constant, and papers always state which one they used:
#
#   "O-rich"  mu_O = E(O2)/2  -- equilibrium with an O2 atmosphere. Implemented here.
#   "O-poor"  mu_O referenced to the bulk oxide's formation enthalpy instead. Gives a
#             systematically different (often >1 eV) E_vac for the same structure.
#
# The label is written into every record so a number can never be read under the wrong
# convention. Adding the O-poor limit means computing the bulk reference and passing a
# different mu_O here; nothing else changes.
OXYGEN_REFERENCE = "O-rich (mu_O = E(O2)/2)"


def oxygen_chemical_potential(
    backend, cfg, relax_fn: Callable, cachedir: str | Path = GAS_CACHE_DIR
) -> float:
    """mu_O in the O-rich limit: half this model's relaxed O2 total energy, in eV.

    Deliberately independent of ``cfg.adsorbate`` — a vacancy run needs oxygen's chemical
    potential whether the adsorbate is O2, CO or nothing at all. Shares the same on-disk
    cache as every other gas reference, so at most one O2 relaxation happens per
    (model, fmax) across an entire sweep.

    Carries the well-known GGA O2 overbinding error, which every PBE-trained model here
    inherits. Because mu_O is *half* E(O2), an error of ~0.5-1 eV in the molecule puts
    ~0.25-0.5 eV directly into every E_vac. It is a constant per model, so trends across
    sites and materials survive it; absolute values should be quoted with the caveat.
    """
    from .config import ADSORBATE_FRAGMENTS

    species, coords = ADSORBATE_FRAGMENTS["O2"]
    e_o2 = gas_reference_energy(backend, cfg, relax_fn, cachedir, species=species, coords=coords)
    return float(e_o2 / 2.0)


# eV; ZPE-corrected experimental formation enthalpy of H2O from H2 + 1/2 O2 (NIST, via
# Kowalski, Meyer & Marx, PRB 79, 115410 (2009), Ref. 61). A reaction energy, not an
# absolute total energy, so it is code/model-independent and safe to mix into any single
# model's own H2/H2O total energies -- unlike an absolute DFT total energy, which is not.
WATER_FORMATION_ENTHALPY_EXP = 2.51


def oxygen_chemical_potential_corrected(
    backend, cfg, relax_fn: Callable, cachedir: str | Path = GAS_CACHE_DIR
) -> float:
    """mu_O via a thermodynamic cycle through this model's own H2 and H2O, instead of its
    raw (possibly overbinding) O2 relaxation.

    Mirrors the trick in Kowalski, Meyer & Marx (2009): GGA-class functionals notoriously
    overbind O2, but describe H2 and H2O well, so solve for E(O2) from the well-behaved
    pair plus the experimental water formation enthalpy instead of trusting the model's
    direct O2 total energy. H2 + 1/2 O2 -> H2O is exothermic by WATER_FORMATION_ENTHALPY_EXP
    (ZPE-corrected experimental Delta_Hf, i.e. E(H2O) - E(H2) - 1/2 E(O2) = -that value), so:

        E(O2)_corrected = 2 * (E(H2O) - E(H2) + WATER_FORMATION_ENTHALPY_EXP)

    E(H2) and E(H2O) come from *this* model, so the whole expression stays on this model's
    own internal energy scale -- only WATER_FORMATION_ENTHALPY_EXP, a reaction energy, is
    external. That is what makes this portable per-model, unlike lifting a literature
    paper's own corrected E(O2) value, which is anchored to that paper's H2/H2O and would
    not cancel correctly against any of these models' slab energies.

    Does not replace ``oxygen_chemical_potential`` (every existing E_vac in this codebase
    keeps using the raw O2 term); this is an opt-in second reference for comparing the two.
    """
    from .config import ADSORBATE_FRAGMENTS

    h2_species, h2_coords = ADSORBATE_FRAGMENTS["H2"]
    h2o_species, h2o_coords = ADSORBATE_FRAGMENTS["H2O"]
    e_h2 = gas_reference_energy(backend, cfg, relax_fn, cachedir, species=h2_species, coords=h2_coords)
    e_h2o = gas_reference_energy(backend, cfg, relax_fn, cachedir, species=h2o_species, coords=h2o_coords)
    e_o2_corrected = 2.0 * (e_h2o - e_h2 + WATER_FORMATION_ENTHALPY_EXP)
    return float(e_o2_corrected / 2.0)


def vacancy_formation_energy(
    e_defective: Optional[float], e_pristine: Optional[float], mu_o: Optional[float]
) -> Optional[float]:
    """E_vac = E(defective slab) - E(pristine slab) + mu_O, in eV.

    Positive; a *smaller* value means the vacancy is easier to form. All three terms must
    come from the same calculator, for the same reason E_ads does: the per-atom energy
    offset only cancels when the atom counts balance across the expression.

    Within one material this is a constant offset from the raw defective-slab energy, so
    it never changes which site the funnel picks. Its value is in making vacancies
    *comparable across materials* — two rutile slabs of different composition have totally
    incomparable total energies, but their formation energies sit on one axis. It is also
    the quantity with a real literature benchmark to check a model against.

    ``None`` if any term is missing, so a stage without a reference state reports nothing
    rather than something wrong.
    """
    if e_defective is None or e_pristine is None or mu_o is None:
        return None
    return float(e_defective - e_pristine + mu_o)
