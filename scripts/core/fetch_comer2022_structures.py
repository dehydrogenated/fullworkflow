"""Fetch Comer et al. 2022's own DFT+U-relaxed bulk rutile structures for all 33 MO2
transition-metal oxides, straight from their public data release (github.com/bencomer/
ComerUnraveling2022, cited as ref. 41 in the paper) -- not Materials Project.

Why: MP only has a genuine spacegroup-136 (rutile) entry for 22 of the 33 -- the other 11
(Ag, Au, Cd, Co, Cu, Ga, Hg, In, Ni, Tl, Zn) don't naturally form rutile and simply aren't
in MP under that spacegroup. Comer et al. computed all 33 anyway, by imposing the rutile
structure type and letting DFT+U relax it -- exactly the convention their published
Delta_E_ads/Delta_G_ads values (data/literature/comer2022_mo2_adsorption/adsorption_
energies.csv) assume. Pulling their literal relaxed cell is more faithful than
reconstructing an approximation via isostructural substitution ourselves.

Written into a folder separate from data/structures/ (which holds our own MP-sourced
working set) -- these are a distinct source with a distinct provenance, and mixing the two
under one naming convention would make it easy to lose track of which is which.

    python scripts/core/fetch_comer2022_structures.py
    python scripts/core/fetch_comer2022_structures.py --formulas AgO2 AuO2

Run once on a networked machine; the pipeline never needs the network again afterward.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

REPO_RAW = "https://raw.githubusercontent.com/bencomer/ComerUnraveling2022/master/bulks"
RUTILE_SPACEGROUP = 136

# formula -> their folder name. 20_LaO2/ScO2/YO2 exist in the source repo too (apparently
# shared with a related paper) but are outside the 33-metal set this benchmark targets.
FOLDERS = {
    "HfO2": "00_HfO2", "TiO2": "00_TiO2", "ZrO2": "00_ZrO2",
    "NbO2": "01_NbO2", "TaO2": "01_TaO2", "VO2": "01_VO2",
    "CrO2": "02_CrO2", "MoO2": "02_MoO2", "WO2": "02_WO2",
    "MnO2": "03_MnO2", "ReO2": "03_ReO2", "TcO2": "03_TcO2",
    "FeO2": "04_FeO2", "OsO2": "04_OsO2", "RuO2": "04_RuO2",
    "CoO2": "05_CoO2", "IrO2": "05_IrO2", "RhO2": "05_RhO2",
    "NiO2": "06_NiO2", "PdO2": "06_PdO2", "PtO2": "06_PtO2",
    "AgO2": "07_AgO2", "AuO2": "07_AuO2", "CuO2": "07_CuO2",
    "CdO2": "08_CdO2", "HgO2": "08_HgO2", "ZnO2": "08_ZnO2",
    "GaO2": "09_GaO2", "InO2": "09_InO2", "TlO2": "09_TlO2",
    "GeO2": "10_GeO2", "PbO2": "10_PbO2", "SnO2": "10_SnO2",
}

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "literature" / \
    "comer2022_mo2_adsorption" / "structures"


def fetch_one(formula: str, folder: str) -> dict:
    url = f"{REPO_RAW}/{folder}/final_with_calculator.json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()

    with tempfile.NamedTemporaryFile(suffix=".json", mode="wb", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    atoms = read(tmp_path)
    Path(tmp_path).unlink()

    structure = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(structure)
    sg_number = sga.get_space_group_number()

    dest = OUT_DIR / formula
    dest.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(dest / f"{formula}.cif"))
    (dest / "final_with_calculator.json").write_bytes(raw)

    calc_params = json.loads(raw)["1"]["calculator_parameters"]
    (dest / "metadata.json").write_text(json.dumps({
        "formula": formula,
        "source": "Comer et al. 2022 data release",
        "source_url": url,
        "citation_doi": "10.1021/acs.jpcc.2c02381",
        "spacegroup_number": sg_number,
        "spacegroup_symbol": sga.get_space_group_symbol(),
        "n_sites": len(structure),
        "energy_eV": atoms.get_potential_energy() if atoms.calc else None,
        "vasp_encut": calc_params.get("encut"),
        "vasp_gga": calc_params.get("gga"),
        "vasp_ispin": calc_params.get("ispin"),
        "vasp_ldau": calc_params.get("ldau"),
    }, indent=2))
    return {"formula": formula, "spacegroup": sg_number, "n_sites": len(structure), "ok": sg_number == RUTILE_SPACEGROUP}


def main(formulas: list[str]) -> None:
    formulas = formulas or list(FOLDERS)
    ok, bad = [], []
    for formula in formulas:
        folder = FOLDERS[formula]
        try:
            res = fetch_one(formula, folder)
        except Exception as e:
            print(f"  {formula:8s} FAILED: {e}")
            bad.append(formula)
            continue
        flag = "" if res["ok"] else f"  ** NOT rutile (spacegroup {res['spacegroup']}) **"
        print(f"  {formula:8s} spacegroup={res['spacegroup']}  n_sites={res['n_sites']}{flag}")
        ok.append(formula)

    print(f"\nwrote {len(ok)}/{len(formulas)} structures to {OUT_DIR}")
    if bad:
        print(f"failed: {', '.join(bad)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--formulas", nargs="+", choices=list(FOLDERS), default=None)
    a = ap.parse_args()
    main(a.formulas or [])
