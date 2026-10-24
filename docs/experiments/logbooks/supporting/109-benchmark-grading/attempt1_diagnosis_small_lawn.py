"""Diagnose the rule's own check on band data: what part is the half-centre's uncertainty."""
import sys, math, numpy as np
sys.path.insert(0,'scripts/analysis')
import benchmark_bands as bb
from quantumnematode.benchmark import grading as g, references as refs
from quantumnematode.validation import roaming_dwelling as rd
tr=refs.load_scheer_tracks()
clf=rd.GaussianClassifier(hmm=rd.load_calibrated_hmm(bb.CLASSIFIER_PATH))
wild=bb.per_date_tallies(tr, bb.WILD_TYPE, clf)
bar=g.z_star(11)
rng=np.random.default_rng(2032); dates=list(wild)
res={'registered':0,'with_centre_var':0}; zs=[]
for _ in range(200):
    o=rng.permutation(len(dates)); h=len(dates)//2
    A={dates[k]:wild[dates[k]] for k in o[:h]}; B=[wild[dates[k]] for k in o[h:]]
    bands=bb.lawn_bands(A, g.LAWN_STATISTICS, 200, rng)
    # the centre's own bootstrap SD: resample A's dates at A's full size
    nA=int(sum(len(v) for v in A.values()))
    cb=g.cluster_bootstrap(list(A.values()), nA, g.lawn_statistics, 400, rng)
    pop=g.cluster_bootstrap(B, 200, g.lawn_statistics, 1, rng); vals={k:float(v[0]) for k,v in pop.items()}
    r1=g.grade(vals,bands,bar); res['registered']+=all(r.passes for r in r1.values())
    b2={k:g.Band(b.centre, math.hypot(b.scale, g.scale_of(cb[k]))) for k,b in bands.items()}
    r2=g.grade(vals,b2,bar); res['with_centre_var']+=all(r.passes for r in r2.values())
    zs.append([r1[k].z for k in g.LAWN_STATISTICS])
zs=np.array(zs)
print({k:v/200 for k,v in res.items()})
print('z SD per statistic (1.0 if the band scale were right):', dict(zip(g.LAWN_STATISTICS, zs.std(axis=0).round(2))))
# date drift: correlation of per-date roaming with date order
per=[g.lawn_statistics(wild[d].sum(axis=0))['roaming_fraction'] for d in dates]
print('per-date roaming, first and last ten dates:', np.round(per[:10],3), np.round(per[-10:],3))
