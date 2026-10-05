# Target selection rule — fixed 2026-10-05 11:47 AEDT, before the target-35 pilot ran

The follow-up panel runs at **the lowest pilot target at which every level (full, gap_held) is
readable** by scripts/campaigns/gate_preflight.py with its default 5-point margin. Gates only; no
wiring gap is read. Targets piloted so far: 25 (saturated), 30 (near the bar), 40 (readable).
Adding 35 on the same seeds (1001-1004): if it is readable it is chosen, otherwise 40.
