# Draft note to the authors of TuragaLab/wormvae

For the maintainer to send, or not, for example as a GitHub issue on the repository or by email.
The draft gives no project details beyond what the report needs.

______________________________________________________________________

**Subject:** Polyadic chemical synapses parsed in reverse in `connectome_preprocess.py`, and a
licence question

Dear authors,

Thank you for releasing the code for "Connectome-constrained latent variable model of whole-brain
neural activity" (ICLR 2022). While reading it at commit `15b58bff`, I found what looks like a
parsing bug, and I have a licence question.

**The parsing.** `model/connectome_preprocess.py`, line 100, treats a row as a send when
`row['Type'] == 'S' or row['Type'] == 'SP'`. In `data/worm_connectivity/white_neuron_connect.csv`
the polyadic send type is spelled `Sp` (1,625 rows). Those rows therefore fall into the `else`
branch and are stored as receives, that is with their direction reversed. Each such synapse also
appears from the partner's side as an `Rp` row, which is stored correctly. By my count:

- 1,270 of the 3,464 chemical entries are reversed copies of real synapses;
- reciprocal chemical pairs rise from about 233 to about 1,503.

If this is right, testing for `'Sp'` (or comparing case-insensitively) would fix it.

**The licence.** The repository has no licence file, so its code cannot be reused. Would you
consider adding one?

With thanks,

[name]
