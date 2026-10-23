"""Familiar vs new programmes in the Top-5 lists (for the business analysis).

For every method at its selected v2 setting, on V1 (history up to week 25 -> weeks 26-30) and on the test
window (history up to week 30 -> weeks 31-42):
  repeat share  = share of the 5 x 1260 slots that the user already watched (>= 50% in one sitting) in the history
  new share     = share of slots the user never overlapped at all in the history (any overlap > 0 s)
  V1 only: precision@5 of repeat slots and of new slots (relevant = watched >= 80% in one sitting in weeks 26-30)
Usage: python src/analysis.py --data data --out v2   (reads the parquet cache in v2/work)
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import prepare
import methods as M

K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--out', default='v2')
    args = ap.parse_args()
    log, sched, A, _ = prepare(args.data, os.path.join(args.out, 'work'))
    tmpl = pd.read_csv(os.path.join(args.data, 'submission.csv'), dtype={'user_id': 'uint64'})
    U = [int(u) for u in tmpl.user_id]
    sel = json.load(open(os.path.join(args.out, 'results', 'summary.json')))['selected']
    rows = []
    for window, t0, tw in (('V1 26-30', 26, range(26, 31)), ('test 31-42', 31, range(31, 43))):
        cache = {}
        for fam, name in sel.items():
            cfg = name.split('[hist=')[1].rstrip(']')
            hw = range(t0 - 10, t0) if cfg == 'last10' else range(11, t0)
            if cfg not in cache:
                cache[cfg] = M.Split(A, log, sched, hw, tw, U, half_life=8 if cfg == 'all_hl8' else None)
            sp_ = cache[cfg]
            p = name.split('(')[1].split(')')[0] if '(' in name else ''
            kv = dict(x.split('=') for x in p.split(', ')) if p else {}
            if fam == 'popularity': lst = M.popularity(sp_)
            elif fam == 'repeat_count': lst = M.repeat_count(sp_)
            elif fam == 'association_rules': lst = M.association_rules(sp_, int(kv['min_support']))[0]
            elif fam == 'ppr': lst = M.ppr(sp_, float(kv['alpha']))[0]
            elif fam == 'als': lst = M.als(sp_, int(kv['factors']), 1.0, 1.0, 15, 42)[0]
            elif fam == 'cbf_metadata': lst = M.cbf_metadata(sp_)[0]
            elif fam == 'cbf_schedule': lst = M.cbf_schedule(sp_)[0]
            else: lst = M.rrf(M.ppr(sp_, float(kv['alpha']))[0], M.cbf_schedule(sp_)[0], float(kv['w_ppr']))
            rec = sp_.finalize(lst, K)
            # familiarity is always judged against the full history up to the window start
            hist = A[A.wk.between(11, t0 - 1)]
            seen50 = set(zip(hist[hist.fmax >= 0.5].user_id, hist[hist.fmax >= 0.5].show))
            touched = set(zip(hist.user_id, hist.show))
            rep = new = hit_rep = hit_new = 0
            for u in U:
                truth = sp_.truth.get(u, set()) if window.startswith('V1') else set()
                for s in rec[u]:
                    if (u, s) in seen50:
                        rep += 1; hit_rep += s in truth
                    elif (u, s) not in touched:
                        new += 1; hit_new += s in truth
            n = K * len(U)
            r = {'window': window, 'method': fam, 'setting': name, 'repeat_share': round(rep / n, 4), 'new_share': round(new / n, 4)}
            if window.startswith('V1'):
                r['precision_repeat_slots'] = round(hit_rep / max(rep, 1), 4)
                r['precision_new_slots'] = round(hit_new / max(new, 1), 4)
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'results', 'novelty.csv'), index=False)


if __name__ == '__main__':
    main()
