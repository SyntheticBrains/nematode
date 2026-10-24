import sys, math, numpy as np
from scipy import stats
sys.path.insert(0,'scripts/analysis')
import benchmark_bands as bb
from quantumnematode.benchmark import grading as g, references as refs
lc=refs.load_lopezcruz(); batches=bb.lopezcruz_batches(lc); rows={r['row']:r for r in lc['rows']}
split=bb.split_batches(batches, rows[90]['dates'])
band=[b for b,s in split['sides'].items() if s=='bands']; st=[bb.search_statistics(batches[b]) for b in band]
print('batches', len(batches), 'band', len(band))
bar=g.z_star(11); rng=np.random.default_rng(2032); reg=tpi=n=0; zs=[]
for _ in range(200):
    o=rng.permutation(len(st)); h=len(st)//2
    A=[st[k] for k in o[:h]]; band_=bb.search_band(A); k=len(A)
    tbar=stats.t.ppf(1-0.05/22, k-1)*math.sqrt(1+1/k)
    for j in o[h:]:
        r=g.grade(st[j], band_, bar); reg+=all(x.passes for x in r.values())
        r2=g.grade(st[j], band_, tbar); tpi+=all(x.passes for x in r2.values()); n+=1
        zs.append([r[s].z for s in g.LOCAL_SEARCH_STATISTICS])
print('registered', reg/n, 't prediction interval', tpi/n, 'z SD', np.array(zs).std(axis=0).round(2))
vals=np.array([[s['gamma_per_min'], s['early_late_ratio']] for s in st]); print('band batches gamma/early-late:', np.round(vals,3).tolist())
