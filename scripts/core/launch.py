"""Interactive launcher — pick a run from menus instead of remembering flags.

    python scripts/core/launch.py

Every prompt lists what is *actually* available right now — materials that resolve
offline, registered adsorbate fragments, registered model heads — so this doubles as the
discovery tool: you never have to grep the registries to find out what you can run.

It prints the equivalent one-line command before starting, so the menus are a way to
learn the flags rather than a replacement for them. Once you know the run you want, copy
that line into a shell script and skip the menus.

Nothing here re-implements the pipeline: it builds a RunConfig from your answers and
calls ``pipeline.run_sweep``, exactly as the flags would.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

from oxide_workflow import pipeline
from oxide_workflow.backends import REGISTRY
from oxide_workflow.config import ADSORBATE_FRAGMENTS, RunConfig
from oxide_workflow.structures import available

# Loose settings that turn a multi-hour run into a few minutes: one adsorbate site per
# position type, a slack force tolerance and a hard step cap. Use it to check that a
# material/facet/adsorbate combination *runs at all* before paying for real numbers.
SMOKE = {"fmax": 0.1, "max_steps": 60, "max_per_position": 1}


def _ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    try:
        return input(f"{prompt}{shown}: ").strip() or default
    except (EOFError, KeyboardInterrupt):
        print("\naborted")
        raise SystemExit(1)


def _pick(title: str, options: list[str], default: int = 0, notes: dict | None = None) -> str:
    """Numbered single-choice menu. Empty input takes the default."""
    print(f"\n{title}")
    for i, opt in enumerate(options):
        note = f"   {notes[opt]}" if notes and opt in notes else ""
        star = " *" if i == default else "  "
        print(f" {i + 1:2d}){star}{opt}{note}")
    while True:
        raw = _ask("choice", str(default + 1))
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  enter 1-{len(options)}")


def _pick_many(title: str, options: list[str], default: list[str]) -> list[str]:
    """Numbered multi-choice menu: "1 3 5", "all", or empty for the default."""
    print(f"\n{title}")
    for i, opt in enumerate(options):
        star = " *" if opt in default else "  "
        print(f" {i + 1:2d}){star}{opt}")
    while True:
        raw = _ask("choices (space-separated, or 'all')", " ".join(
            str(options.index(d) + 1) for d in default if d in options
        ))
        if raw.lower() == "all":
            return list(options)
        parts = raw.split()
        if parts and all(p.isdigit() and 1 <= int(p) <= len(options) for p in parts):
            return [options[int(p) - 1] for p in dict.fromkeys(parts)]
        print(f"  enter numbers 1-{len(options)}, or 'all'")


class _Tee:
    """Mirror stdout to a log file, so a long run leaves a transcript without ``| tee``."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.log = path.open("w")

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()  # a run can take hours; keep the log readable while it runs

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def _command(cfg: RunConfig, adsorbate: str, candidates: list[str], protocol: str,
             outdir: str, smoke: bool) -> str:
    """The equivalent one-liner — printed so the menus teach the flags.

    ``adsorbate`` is the fragment *key*, which is what --adsorbate takes; joining the
    species instead would emit an unusable "OO"/"OHH".
    """
    parts = [
        "python -m oxide_workflow.pipeline",
        f"--material {cfg.polymorph}",
        f"--miller {''.join(map(str, cfg.slab.miller_index))}",
        f"--adsorbate {adsorbate}",
        f"--protocol {protocol}",
        f"--candidates {' '.join(candidates)}",
        f"--outdir {outdir}",
    ]
    if smoke:
        parts += [f"--fmax {cfg.relax.fmax}", f"--max-steps {cfg.relax.max_steps}",
                  f"--max-sites {cfg.adsorbate.max_per_position}"]
    return " \\\n    ".join(parts)


def main() -> None:
    base = RunConfig()
    print("=" * 70)
    print("oxide-workflow launcher — '*' marks the default, Enter accepts it")
    print("=" * 70)

    material = _pick("MATERIAL (resolves offline)", available(),
                     default=max(available().index(base.polymorph), 0)
                     if base.polymorph in available() else 0)

    miller = _ask("\nFACET as '110' or '1,1,-1'",
                  "".join(map(str, base.slab.miller_index)))

    adsorbate = _pick(
        "ADSORBATE fragment", sorted(ADSORBATE_FRAGMENTS),
        default=sorted(ADSORBATE_FRAGMENTS).index("CO"),
        notes={
            "O2": "physical O reservoir; carries GGA O2 overbinding error",
            "O2-side": "flat start — use when you expect dissociation at a vacancy",
            "O": "atomic; vacancy healing, but the shakiest gas reference",
        },
    )

    protocol = _pick(
        "PROTOCOL", ["full_pipeline", "seeded", "both"], default=0,
        notes={
            "full_pipeline": "candidate's own chain — realistic accumulated error",
            "seeded": "each stage from the reference — per-stage error, for DFT comparison",
            "both": "roughly double the candidate cost",
        },
    )

    reference = _pick("REFERENCE model (the ground truth)", sorted(REGISTRY),
                      default=sorted(REGISTRY).index(base.reference))
    candidates = _pick_many(
        "CANDIDATE model(s) to benchmark against it",
        [m for m in sorted(REGISTRY) if m != reference],
        default=[base.candidate] if base.candidate != reference else [],
    )
    if not candidates:
        raise SystemExit("no candidates chosen — nothing to benchmark")

    speed = _pick(
        "SPEED", ["full", "smoke test"], default=0,
        notes={
            "full": "every site, tight convergence — hours",
            "smoke test": f"1 site/position, fmax={SMOKE['fmax']}, "
                          f"{SMOKE['max_steps']} steps — minutes, numbers NOT publishable",
        },
    )
    smoke = speed == "smoke test"

    stamp = date.today().strftime("%m%d")
    suggested = f"runs/{adsorbate}_{material}_{'smoke' if smoke else protocol}_{stamp}"
    outdir = Path(_ask("\nOUTPUT directory", suggested))

    # ---- assemble the config exactly as the flags would --------------------------------
    species, coords = ADSORBATE_FRAGMENTS[adsorbate]
    cfg = replace(
        base,
        polymorph=material,
        reference=reference,
        slab=replace(base.slab, miller_index=pipeline._cli_miller(miller)),
        adsorbate=replace(base.adsorbate, species=species, coords=coords),
    )
    if smoke:
        cfg = replace(
            cfg,
            relax=replace(cfg.relax, fmax=SMOKE["fmax"], max_steps=SMOKE["max_steps"]),
            adsorbate=replace(cfg.adsorbate, max_per_position=SMOKE["max_per_position"]),
        )
    protocols = pipeline.PROTOCOLS if protocol == "both" else (protocol,)

    # ---- confirm -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"material    {material}   facet {miller}")
    print(f"adsorbate   {adsorbate}  ({'-'.join(species)})")
    print(f"reference   {reference}")
    print(f"candidates  {', '.join(candidates)}")
    print(f"protocols   {', '.join(protocols)}")
    print(f"relax       fmax={cfg.relax.fmax}  max_steps={cfg.relax.max_steps}"
          + (f"  max_sites={cfg.adsorbate.max_per_position}" if smoke else ""))
    print(f"outdir      {outdir}")
    print("\nequivalent command:\n")
    print(_command(cfg, adsorbate, candidates, protocol, str(outdir), smoke))
    print("=" * 70)

    if outdir.exists() and any(outdir.iterdir()):
        print(f"\n!! {outdir} already exists and is not empty.")
        print("   The run will truncate divergence.jsonl / candidates.jsonl in it.")
        if _ask("   overwrite? (yes/no)", "no").lower() not in ("y", "yes"):
            raise SystemExit("cancelled")

    if _ask("\nstart? (yes/no)", "yes").lower() not in ("y", "yes"):
        raise SystemExit("cancelled")

    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir.with_suffix(".log")
    tee = _Tee(sys.stdout, log_path)
    sys.stdout = tee
    try:
        print(f"\nlogging to {log_path}\n")
        summary = pipeline.run_sweep(
            candidates, cfg=cfg, outdir=outdir, protocols=protocols
        )
        print(f"\ndone in {summary['total_elapsed_s'] / 60:.1f} min")
        print(f"min E_ads (reference): {summary.get('reference_min_e_ads')} eV")
        for name, d in summary["per_candidate"].items():
            print(f"min E_ads ({name}): {d.get('min_e_ads')} eV")
        print(f"\nreport:  python scripts/core/report.py {outdir}")
    except KeyboardInterrupt:
        print("\ninterrupted — partial results are on disk; report.py reads them")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
