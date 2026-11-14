# Launch records for Logbook 134

Both readings in Logbook 134 were registered, and committed, before they were computed. Each was
computed once. The registrations live with the investigation that made them. They are listed here
with the commits that fix their order:

| Reading | Registration | Registered | Amended | Computed |
|---|---|---|---|---|
| M-a: C.2 against Creamer et al.'s measured responses | [investigation/spike/MA_C2_READING.md](../../../../../investigation/spike/MA_C2_READING.md) | 892a0105 | d9f928ac, before any measured matrix was loaded | b401f07c |
| H0: the true wiring against 19 degree-preserving rewirings | [investigation/spike/H0_READING.md](../../../../../investigation/spike/H0_READING.md) | edfafa94 | none | 6ff2b2d5 |

The commit hashes are those of the investigation branch, `investigation/fit-first`.

M-a's amendment corrected two registered rules that were wrong about the code:

- the answer vector's layout;
- how stimulation enters the encoded forward.

The parity check had already run, and no measured data had been read.
