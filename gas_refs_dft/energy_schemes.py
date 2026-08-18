#!/usr/bin/env python3
"""
The four energy expressions you are comparing, written out explicitly so the
difference between schemes is a one-line change, not a bug.

Convention used here (yours, not OC22's - see README):
    MIXED-ML : E_sys and E_slab from the MLIP, E_gas from DFT
    FULL-ML  : E_sys, E_slab AND E_gas all from the MLIP
"""


# ===========================================================================
# ADSORPTION ENERGY
# ===========================================================================
def e_ads(e_sys, e_slab, e_gas, n_ads=1):
    """
    E_ads = [ E_sys - E_slab - n_ads * E_gas ] / n_ads

    Identical algebra in both schemes. The ONLY difference is where e_gas
    comes from. Everything else must come from the same source, and e_slab
    must be the slab the adsorbate actually sits on (see README on
    consistent references).
    """
    return (e_sys - e_slab - n_ads * e_gas) / n_ads


# ===========================================================================
# OXYGEN VACANCY FORMATION ENERGY  (the MvK step - read the README warning)
# ===========================================================================
def e_vac_O2(e_slab_vac, e_slab, e_O2):
    """
    E_vac = E_slab-with-Ovac + 1/2 E(O2) - E_slab

    Direct O2 reference. Simple, and WRONG by a known, large amount in PBE
    (O2 overbinding, plus you are mixing a GGA molecule with a GGA+U oxide).
    Use for internal trends only, never for absolute numbers.
    """
    return e_slab_vac + 0.5 * e_O2 - e_slab


def e_vac_H2O(e_slab_vac, e_slab, e_H2O, e_H2, dG_corr=2.46):
    """
    E_vac = E_slab-with-Ovac + [E(H2O) - E(H2) + dG_corr] - E_slab

    Norskov's standard trick: replace the badly-described O2 with the
    well-described H2O/H2 pair, and absorb the difference into an
    experimentally-fitted constant.

        1/2 O2  ==  H2O - H2 + 2.46 eV   (PBE, 298 K, 0.035 bar H2O)

    This is the reference the surface-science literature actually uses, and
    it is what the Norskov chapter in your project describes. Strongly
    preferred over e_vac_O2 for anything you intend to publish.
    """
    return e_slab_vac + (e_H2O - e_H2 + dG_corr) - e_slab


# ===========================================================================
# CORRECTED FULL-ML
# ===========================================================================
def e_ads_full_ml_corrected(e_sys_ml, e_slab_ml, e_gas_ml, delta_gas, n_ads=1):
    """
    Full-ML with the measured per-species gas offset subtracted back out.

        delta_gas = E_gas^MLIP - E_gas^DFT   (from gas_offset_diagnostic.py)

    This recovers most of Mixed-ML's accuracy at Full-ML's cost, because the
    offset is roughly constant per molecule and transfers across surfaces.
    You still need the DFT gas calculation ONCE to measure delta_gas - but
    only once, not per functional-per-surface.
    """
    e_gas_corrected = e_gas_ml - delta_gas
    return (e_sys_ml - e_slab_ml - n_ads * e_gas_corrected) / n_ads


# ===========================================================================
# ERROR DECOMPOSITION - what you should see if everything is wired correctly
# ===========================================================================
def error_budget(e_ads_ml, e_ads_dft):
    """
    Run this over a DFT-validated subset (10-20 systems is enough).

    Diagnostics to check:
      1. Does the error grow with slab size?
         YES -> your slab error is NOT cancelling. Suspect an inconsistent
                slab reference: you relaxed the clean slab and the
                adsorbate+slab independently and they found different minima.
         NO  -> cancellation is working.

      2. Is the error a constant shift per adsorbate species?
         YES -> that is the gas/intramolecular reference offset. Correctable.
         NO  -> genuine adsorbate-surface bond error. Not correctable; this
                is the floor of what the MLIP can do for you.
    """
    import numpy as np
    err = np.asarray(e_ads_ml) - np.asarray(e_ads_dft)
    return {
        "ME (systematic shift)": float(err.mean()),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE after removing shift": float(np.abs(err - err.mean()).mean()),
    }
