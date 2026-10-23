"""End-to-end run: data preparation, validation of every method on the same splits, model selection,
final Top-5 submissions for weeks 31-42, examples for two users. Usage:
    python src/run_all.py --data data --out .
"""
import argparse, json, os, platform, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import prepare
from metrics import mapk, slide_example
import methods as M

K = 5
SEED = 42
GRID = {'association_rules': [2, 5, 20], 'ppr': [0.5, 0.8, 0.9], 'als': [16, 64], 'hybrid_w_ppr': [0.3, 0.5, 0.7]}


def log_print(fh, *a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    fh.write(s + '\n'); fh.flush()


def time_offset_check(log, sched, n=300000, seed=0):
    """Share of view start times within 60 s of a programme start on the same channel, for clock offsets."""
    samp = log.sample(n, random_state=seed)
    res = {}
    for off in (-2, -1, 0, 1, 2):
        d_all = []
        st_all = sched.start + pd.Timedelta(hours=off)
        for ch, g in samp.groupby('vsetv_id'):
            ss = np.sort(st_all[sched.channel_id == ch].values)
            if len(ss) == 0:
                continue
            v = g.start.values
            i = np.searchsorted(ss, v)
            a, b = ss[np.clip(i - 1, 0, len(ss) - 1)], ss[np.clip(i, 0, len(ss) - 1)]
            d_all.append(np.minimum(np.abs(v - a), np.abs(b - v)) / np.timedelta64(1, 's'))
        res[off] = round(float((np.concatenate(d_all) <= 60).mean()), 4)
    return res


def run_methods(sp_, fh, tag, timings):
    """All methods (and their small grids) on one split. Returns {name: lists} and {name: MAP@5}."""
    lists, scores = {}, {}
    eval_users = [u for u in sp_.users if u in sp_.truth]

    def add(name, lst, t):
        lists[name] = lst
        scores[name] = mapk(sp_.truth, sp_.finalize(lst, K), eval_users, K)
        timings.setdefault(name, {})[tag] = round(time.time() - t, 1)
        log_print(fh, f'  [{tag}] {name:28s} MAP@5 = {scores[name]:.4f}   ({timings[name][tag]:.1f}s)')

    t = time.time(); add('popularity', M.popularity(sp_), t)
    t = time.time(); add('repeat_count (reference)', M.repeat_count(sp_), t)
    for ms in GRID['association_rules']:
        t = time.time(); add(f'association_rules(min_support={ms})', M.association_rules(sp_, ms)[0], t)
    for a in GRID['ppr']:
        t = time.time(); add(f'ppr(alpha={a})', M.ppr(sp_, a)[0], t)
    for f in GRID['als']:
        t = time.time(); add(f'als(factors={f})', M.als(sp_, f, 1.0, 1.0, 15, SEED)[0], t)
    t = time.time(); add('cbf_metadata', M.cbf_metadata(sp_)[0], t)
    t = time.time(); add('cbf_schedule', M.cbf_schedule(sp_)[0], t)
    return lists, scores, eval_users


def best_of(scores, prefix):
    c = {k: v for k, v in scores.items() if k.startswith(prefix)}
    return max(c, key=lambda k: (c[k], k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--out', default='.')
    args = ap.parse_args()
    res_dir, sub_dir, work = (os.path.join(args.out, d) for d in ('results', 'submissions', 'work'))
    for d in (res_dir, sub_dir):
        os.makedirs(d, exist_ok=True)
    fh = open(os.path.join(res_dir, 'run_log.txt'), 'w', encoding='utf-8')
    T0 = time.time()
    env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'machine': platform.machine(),
           'cpu_count': os.cpu_count(), 'numpy': np.__version__, 'pandas': pd.__version__}
    import scipy; env['scipy'] = scipy.__version__
    log_print(fh, 'environment', json.dumps(env))
    x, y = slide_example()
    log_print(fh, f'metric self-check (course slide example): X = {x:.4f} (slide 0.917), Y = {y:.4f} (slide 0.533)')

    # ---------------- 1. data
    log, sched, A, stats = prepare(args.data, work)
    log_print(fh, 'data', json.dumps(stats, ensure_ascii=False))
    tmpl = pd.read_csv(os.path.join(args.data, 'submission.csv'), dtype={'user_id': 'uint64'})
    U = [int(u) for u in tmpl.user_id]
    stats['test_users'] = len(U)
    stats['test_users_in_log'] = int(len(set(U) & set(log.user_id.unique().tolist())))
    stats['time_offset_check_share_within_60s'] = time_offset_check(log, sched)
    log_print(fh, 'clock-offset check (share of view starts within 60 s of a programme start):', stats['time_offset_check_share_within_60s'])
    thr = {}
    for name, m in [('overlap>=60s', A.ov >= 60), ('overlap>=300s', A.ov >= 300), ('single sitting>=30%', A.fmax >= 0.3),
                    ('single sitting>=50% (history)', A.fmax >= 0.5), ('single sitting>=80% (label)', A.fmax >= 0.8)]:
        thr[name] = {'user_airings': int(m.sum()), 'user_programme_pairs': int(A[m].groupby(['user_id', 'show']).ngroups)}
    stats['threshold_table'] = thr
    stats['view_duration_quantiles_s'] = {str(q): float(log.duraton.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
    stats['airing_fmax_quantiles'] = {str(q): round(float(A.fmax.quantile(q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    log_print(fh, 'thresholds', json.dumps(thr))

    # ---------------- 2. validation (same pool rule, fallback, K and users for every method)
    splits = {'V0 (hist 11-20 -> val 21-25)': (range(11, 21), range(21, 26)),
              'V1 (hist 16-25 -> val 26-30)': (range(16, 26), range(26, 31))}
    table, timings, keep = {}, {}, {}
    for tag, (hw, tw) in splits.items():
        t = time.time()
        sp_ = M.Split(A, log, sched, hw, tw, U)
        log_print(fh, f'{tag}: pool {len(sp_.pool)} programmes, users with history {len(sp_.eu)}/{len(U)}, '
                      f'users with >=1 relevant {sum(1 for u in U if u in sp_.truth)}, setup {time.time()-t:.1f}s')
        lists, scores, eval_users = run_methods(sp_, fh, tag.split()[0], timings)
        best_ppr = best_of(scores, 'ppr'); ppr_lists = lists[best_ppr]
        for w in GRID['hybrid_w_ppr']:
            t = time.time()
            name = f'hybrid RRF(ppr, cbf_schedule; w_ppr={w})'
            lists[name] = M.rrf(ppr_lists, lists['cbf_schedule'], w)
            scores[name] = mapk(sp_.truth, sp_.finalize(lists[name], K), eval_users, K)
            log_print(fh, f'  [{tag.split()[0]}] {name:28s} MAP@5 = {scores[name]:.4f}  (uses {best_ppr})')
        table[tag] = scores
        keep[tag] = (sp_, lists)
    V1 = 'V1 (hist 16-25 -> val 26-30)'
    sel = {'association_rules': best_of(table[V1], 'association_rules'), 'ppr': best_of(table[V1], 'ppr'),
           'als': best_of(table[V1], 'als'), 'hybrid': best_of(table[V1], 'hybrid')}
    log_print(fh, 'selected on V1:', json.dumps(sel))
    vt = pd.DataFrame(table).rename_axis('method').reset_index()
    vt.to_csv(os.path.join(res_dir, 'validation_map5.csv'), index=False, float_format='%.4f')

    # ---------------- 3. final model: history 21-30 -> test weeks 31-42
    t = time.time()
    F = M.Split(A, log, sched, range(21, 31), range(31, 43), U)
    log_print(fh, f'FINAL (hist 21-30 -> test 31-42): pool {len(F.pool)}, users with history {len(F.eu)}/{len(U)} '
                  f'(the other {len(U) - len(F.eu)} get the popularity list), setup {time.time()-t:.1f}s')
    ms_ar = int(sel['association_rules'].split('=')[1].rstrip(')'))
    a_ppr = float(sel['ppr'].split('=')[1].rstrip(')'))
    w_h = float(sel['hybrid'].split('=')[1].rstrip(')'))
    fl = {}
    t = time.time(); fl['popularity'] = M.popularity(F)
    fl['repeat_count'] = M.repeat_count(F)
    fl['association_rules'] = M.association_rules(F, ms_ar)[0]
    fl['ppr'] = M.ppr(F, a_ppr)[0]
    f_als = int(sel['als'].split('=')[1].rstrip(')'))
    fl['als'] = M.als(F, f_als, 1.0, 1.0, 15, SEED)[0]
    fl['cbf_metadata'] = M.cbf_metadata(F)[0]
    fl['cbf_schedule'] = M.cbf_schedule(F)[0]
    fl['hybrid'] = M.rrf(fl['ppr'], fl['cbf_schedule'], w_h)
    log_print(fh, f'final lists for all methods in {time.time()-t:.1f}s')
    checks = {}
    for name, lst in fl.items():
        rec = F.finalize(lst, K)
        out = tmpl[['user_id']].copy()
        out['tv_show_id'] = [' '.join(str(s) for s in rec[u]) for u in U]
        path = os.path.join(sub_dir, f'submission_{name}.csv')
        out.to_csv(path, index=False)
        ok = (len(out) == len(tmpl) and out.user_id.tolist() == tmpl.user_id.tolist()
              and all(len(set(r)) == K and 0 not in r and all(s in set(F.pool) for s in r) for r in rec.values()))
        checks[name] = ok
        log_print(fh, f'  wrote {path}: rows {len(out)}, format and pool check {"OK" if ok else "FAILED"}')
    stats['submission_checks'] = checks

    # ---------------- 4. two example users (same users for every method)
    sp1, l1 = keep[V1]
    hist_n = sp1.H[sp1.H.user_id.isin(set(U))].groupby('user_id').size()
    cand = hist_n[[u in sp1.truth for u in hist_n.index]]
    heavy = int(cand.idxmax()); light = int(cand[cand >= 3].idxmin())
    meta = sched[sched.tv_show_id != 0].drop_duplicates('tv_show_id', keep='last').set_index('tv_show_id')
    rows = []
    name_map = {'popularity': 'popularity', 'association_rules': sel['association_rules'], 'ppr': sel['ppr'],
                'als': sel['als'], 'cbf_metadata': 'cbf_metadata', 'cbf_schedule': 'cbf_schedule', 'hybrid': sel['hybrid']}
    for role, u in (('heavy', heavy), ('light', light)):
        for mname, vname in name_map.items():
            for window, ids, truth in (('V1 26-30', sp1.finalize(l1[vname], K)[u], sp1.truth.get(u, set())),
                                       ('test 31-42', F.finalize(fl[mname], K)[u], None)):
                for r, s in enumerate(ids, 1):
                    rows.append({'user_role': role, 'user_id': str(u), 'window': window, 'method': mname, 'rank': r, 'tv_show_id': s,
                                 'title': meta.tv_show_title.get(s, ''), 'category': meta.tv_show_category.get(s, ''),
                                 'watched_in_window_ge80': (s in truth) if truth is not None else '',
                                 'times_watched_in_history_ge50': int(((sp1.H.user_id == u) & (sp1.H.show == s)).sum()) if window.startswith('V1')
                                 else int(((F.H.user_id == u) & (F.H.show == s)).sum())})
        hist = sp1.H[sp1.H.user_id == u].groupby('show').size().sort_values(ascending=False).head(10)
        stats[f'example_{role}'] = {'user_id': str(u), 'history_airings_ge50_weeks16_25': int(hist_n[u]),
                                    'distinct_programmes_history': int(sp1.H[sp1.H.user_id == u].show.nunique()),
                                    'relevant_in_26_30': len(sp1.truth.get(u, set())),
                                    'top_history': [[int(s), meta.tv_show_title.get(s, ''), meta.tv_show_category.get(s, ''), int(c)] for s, c in hist.items()]}
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, 'examples_two_users.csv'), index=False)
    stats['selected'] = sel
    stats['timings_s'] = timings
    stats['total_runtime_s'] = round(time.time() - T0, 1)
    stats['environment'] = env
    json.dump(stats, open(os.path.join(res_dir, 'summary.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    log_print(fh, f'done in {stats["total_runtime_s"]} s; results in {res_dir}, submissions in {sub_dir}')


if __name__ == '__main__':
    main()
