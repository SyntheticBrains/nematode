# Draft note to the atlas authors

For the maintainer to send, or not. Check the corresponding author on the paper and the DANDI record
before sending. The draft gives no project details beyond what the request needs.

______________________________________________________________________

**Subject:** Stimulation target identities in DANDI 001075 (Randi et al. 2023 atlas)

Dear Professor Leifer,

Thank you for releasing the signal propagation atlas on DANDI under CC-BY. I am using it to
reproduce the connectome-constrained LDS fit of Creamer, Leifer and Pillow (bioRxiv
2024.09.22.614271), and I have run into a gap that you may be able to close.

In the 110 segmentation files of DANDI 001075 (version 0.240930.1859), the
`OptogeneticStimulusTable` gives `target_pumpprobe_id` for 2,815 of 5,614 stimulations. In 52 files
fewer than half of the stimulations have a target identity, and in 2 none do. Matching the stored
target coordinates to the segmented neurons recovers the known identities in only about 56 to 65%
of cases. The NeuroPAL labels also do not reproduce the 154-neuron set used by Creamer et al.

Creamer et al.'s measured response arrays contain stimulated pairs that the identified DANDI
stimulations cannot produce. So it looks as though a fuller, curated record of targets and labels
exists, perhaps the exports on OSF.

Would either of these be possible?

1. Adding the full target identities and curated labels to the DANDI deposit.
2. Stating a licence, such as CC-BY, on the OSF exports, so they can be reused on the same terms as
   the DANDI data.

I would be glad to share the per-file counts if they would help.

With thanks,

[name]
