"""diverge(struct_A, struct_B) -> DivergenceRecord.

The displacement triple (mean, rmsd, max) comes from a *single* per-atom displacement
vector, computed after StructureMatcher alignment under PBC:

- ``rmsd ≈ mean``  ⇒ uniform drift (whole cell shifted/breathed)
- ``rmsd ≫ mean``  ⇒ localized failure (one region moved a lot)
- ``max_disp_atom`` / ``max_disp_species`` name the culprit.

``energy_error`` is the candidate energy at *its own* minimum minus the reference
energy. ``start_fmax_at_ref_geom`` (force felt at the stage input before relaxing) is
supplied by the caller from the relaxation, not recomputed here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from .records import DivergenceRecord

# scale=False: we want physical displacements in Å, never volume-normalized.
_DEFAULT_MATCHER = StructureMatcher(
    primitive_cell=False, scale=False, attempt_supercell=False
)


def per_atom_displacements(
    ref: Structure, cand: Structure, matcher: StructureMatcher | None = None
) -> Optional[np.ndarray]:
    """Per-atom displacement magnitudes (Å) of ``cand`` vs ``ref`` after PBC alignment.

    Returns an array ordered like ``ref`` sites, or ``None`` if the matcher cannot map
    the two structures (a divergence so large the topology no longer corresponds).
    """
    matcher = matcher or _DEFAULT_MATCHER
    cand_like = matcher.get_s2_like_s1(ref, cand)
    if cand_like is None:
        return None
    frac = cand_like.frac_coords - ref.frac_coords
    frac -= np.round(frac)  # minimum image under PBC
    cart = frac @ ref.lattice.matrix
    return np.linalg.norm(cart, axis=1)


def diverge(
    ref: Structure,
    cand: Structure,
    *,
    stage: str,
    model: str,
    e_ref: Optional[float] = None,
    e_cand: Optional[float] = None,
    start_fmax: Optional[float] = None,
    matcher: StructureMatcher | None = None,
    polymorph: str = "",
    facet: str = "",
    termination: str = "",
    geometry_source: str = "",
    protocol: str = "",
    active_site_dBO: Optional[float] = None,
    symmetry_match: Optional[bool] = None,
    **meta,
) -> DivergenceRecord:
    """Compare a candidate relaxed structure against the reference for one stage.

    ``ref`` / ``cand`` are relaxed structures. Energies (optional) yield
    ``energy_error = e_cand - e_ref``. Extra ``meta`` kwargs are stored on the record.
    """
    d = per_atom_displacements(ref, cand, matcher)

    if d is None:
        rec_meta = {"matched": False, **meta}
        mean = rmsd = maxd = None
        max_atom = max_species = None
    else:
        rec_meta = {"matched": True, **meta}
        mean = float(d.mean())
        rmsd = float(np.sqrt((d ** 2).mean()))
        maxd = float(d.max())
        max_atom = int(np.argmax(d))
        max_species = str(ref[max_atom].specie)

    energy_error = None
    if e_ref is not None and e_cand is not None:
        energy_error = float(e_cand - e_ref)

    return DivergenceRecord(
        composition=ref.composition.reduced_formula,
        stage=stage,
        model=model,
        polymorph=polymorph,
        facet=facet,
        termination=termination,
        geometry_source=geometry_source,
        protocol=protocol,
        start_fmax_at_ref_geom=start_fmax,
        mean_displacement=mean,
        rmsd=rmsd,
        max_displacement=maxd,
        max_disp_atom=max_atom,
        max_disp_species=max_species,
        energy_error=energy_error,
        active_site_dBO=active_site_dBO,
        symmetry_match=symmetry_match,
        meta=rec_meta,
    )
