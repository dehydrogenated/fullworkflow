"""Oxide surface reactivity workflow — MLIP-vs-reference divergence benchmarking.

Summary:
pipeline.py asks structures.py for a bulk crystal. It sends it to backends.py to be relaxed 
(which farms it out to worker_relax.py). It hands the relaxed bulk to stages.py to cut a slab, 
sends that off to be relaxed, hands that to stages.py to make 6 vacancy structures, relaxes all 6, 
keeps the best, hands it to stages.py for 7 adsorbate placements, relaxes all 7. Then it does the 
same chain with the candidate model, and uses diverge.py to compare the two at every step. 
checks.py flags anything suspicious, records.py writes it all down, and config.py supplied 
the numbers throughout.
"""

__version__ = "0.3.0" # July 28th
