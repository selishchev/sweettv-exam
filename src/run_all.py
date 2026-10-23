"""End-to-end run (v2): data preparation, validation of every method on the same splits and three history
configurations, model selection on V1, final Top-5 submissions for weeks 31-42, examples for two users. Usage:
    python src/run_all.py --data data --out v2
"""
import argparse, gc, json, os, platform, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep import prepare
from metrics import mapk, slide_example
import methods as M

K = 5
SEED = 42
GRID = {'association_rules': [2, 5, 20], 'ppr': [0.5, 0.8, 0.9], 'als': [16, 64], 'hybrid_w_ppr': [0.5, 0.7, 0.85]}
# history window: the last 10 weeks before the target window, or all weeks from week 11; optional recency half-life (weeks)
HIST_CFGS = {'last10': None, 'all': None, 'all_hl8': 8}


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


def run_methods(sp_, fh, tag, cfg, timings, reg):
    """All methods (and their small grids) on one split and one history configuration."""
    lists, scores = {}, {}
    eval_users = [u for u in sp_.users if u in sp_.truth]

    def add(family, params, lst, t):
        name = f'{family}({params}) [hist={cfg}]' if params else f'{family} [hist={cfg}]'
        reg[name] = (family, cfg, params)
        lists[name] = lst
        scores[name] = mapk(sp_.truth, sp_.finalize(lst, K), eval_users, K)
        timings.setdefault(name, {})[tag] = round(time.time() - t, 1)
        log_print(fh, f'  [{tag}] {name:48s} MAP@5 = {scores[name]:.4f}   ({timings[name][tag]:.1f}s)')
        return name

    t = time.time(); add('popularity', '', M.popularity(sp_), t)
    t = time.time(); add('repeat_count', '', M.repeat_count(sp_), t)
    for ms in GRID['association_rules']:
        t = time.time(); add('association_rules', f'min_support={ms}', M.association_rules(sp_, ms)[0], t)
    best_ppr, best_s = None, -1
    for a in GRID['ppr']:
        t = time.time(); n = add('ppr', f'alpha={a}', M.ppr(sp_, a)[0], t)
        if scores[n] > best_s:
            best_ppr, best_s = n, scores[n]
    for f in GRID['als']:
        t = time.time(); add('als', f'factors={f}', M.als(sp_, f, 1.0, 1.0, 15, SEED)[0], t)
    t = time.time(); add('cbf_metadata', '', M.cbf_metadata(sp_)[0], t)
    t = time.time(); cs = add('cbf_schedule', '', M.cbf_schedule(sp_)[0], t)
    a_best = reg[best_ppr][2]
    for w in GRID['hybrid_w_ppr']:
        t = time.time(); add('hybrid', f'{a_best}, w_ppr={w}', M.rrf(lists[best_ppr], lists[cs], w), t)
    return lists, scores


def history_weeks(cfg, target_start):
    return range(target_start - 10, target_start) if cfg.startswith('last10') else range(11, target_start)


def best_in(scores, reg, family):
    c = {k: v for k, v in scores.items() if reg[k][0] == family}
    return max(c, key=lambda k: (c[k], k))


def param(params, key, cast):
    for part in params.split(', '):
        k, v = part.split('=')
        if k == key:
            return cast(v)
    raise KeyError(key)


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
    splits = {'V0': ('V0 (val 21-25)', 21, range(21, 26)), 'V1': ('V1 (val 26-30)', 26, range(26, 31))}
    table, timings, keep, reg = {}, {}, {}, {}
    for tag, (label, t0w, tw) in splits.items():
        table[label] = {}
        for cfg, hl in HIST_CFGS.items():
            hw = history_weeks(cfg, t0w)
            t = time.time()
            sp_ = M.Split(A, log, sched, hw, tw, U, half_life=hl)
            log_print(fh, f'{label}, history weeks {hw.start}-{hw.stop - 1} [{cfg}]: pool {len(sp_.pool)} programmes, '
                          f'users with history {len(sp_.eu)}/{len(U)}, users with >=1 relevant {sum(1 for u in U if u in sp_.truth)}, '
                          f'setup {time.time()-t:.1f}s')
            lists, scores = run_methods(sp_, fh, tag, cfg, timings, reg)
            table[label].update(scores)
            sp_.Aall = None                       # free memory: only H (watched >= 50%) is needed later
            if tag == 'V1':
                keep[(tag, cfg)] = (sp_, lists)
            del sp_, lists
            gc.collect()
    V1 = splits['V1'][0]
    families = ['popularity', 'repeat_count', 'association_rules', 'ppr', 'als', 'cbf_metadata', 'cbf_schedule', 'hybrid']
    sel = {f: best_in(table[V1], reg, f) for f in families}
    log_print(fh, 'selected on V1:', json.dumps(sel))
    vt = pd.DataFrame(table).rename_axis('method').reset_index()
    vt.to_csv(os.path.join(res_dir, 'validation_map5.csv'), index=False, float_format='%.4f')

    # ---------------- 3. final model: all history up to week 30 (or the selected window) -> test weeks 31-42
    finals, fl = {}, {}
    t = time.time()
    for fam in families:
        _, cfg, params = reg[sel[fam]]
        if cfg not in finals:
            hw = history_weeks(cfg, 31)
            finals[cfg] = M.Split(A, log, sched, hw, range(31, 43), U, half_life=HIST_CFGS[cfg])
            log_print(fh, f'FINAL [{cfg}] history weeks {hw.start}-{hw.stop - 1} -> test 31-42: pool {len(finals[cfg].pool)}, '
                          f'users with history {len(finals[cfg].eu)}/{len(U)} (others get the popularity list)')
        F = finals[cfg]
        if fam == 'popularity':
            fl[fam] = M.popularity(F)
        elif fam == 'repeat_count':
            fl[fam] = M.repeat_count(F)
        elif fam == 'association_rules':
            fl[fam] = M.association_rules(F, param(params, 'min_support', int))[0]
        elif fam == 'ppr':
            fl[fam] = M.ppr(F, param(params, 'alpha', float))[0]
        elif fam == 'als':
            fl[fam] = M.als(F, param(params, 'factors', int), 1.0, 1.0, 15, SEED)[0]
        elif fam == 'cbf_metadata':
            fl[fam] = M.cbf_metadata(F)[0]
        elif fam == 'cbf_schedule':
            fl[fam] = M.cbf_schedule(F)[0]
        elif fam == 'hybrid':
            p_l = M.ppr(F, param(params, 'alpha', float))[0]
            fl[fam] = M.rrf(p_l, M.cbf_schedule(F)[0], param(params, 'w_ppr', float))
    log_print(fh, f'final lists for all methods in {time.time()-t:.1f}s')
    checks = {}
    for name, lst in fl.items():
        F = finals[reg[sel[name]][1]]
        rec = F.finalize(lst, K)
        out = tmpl[['user_id']].copy()
        out['tv_show_id'] = [' '.join(str(s) for s in rec[u]) for u in U]
        path = os.path.join(sub_dir, f'submission_{name}.csv')
        out.to_csv(path, index=False)
        ok = (len(out) == len(tmpl) and out.user_id.tolist() == tmpl.user_id.tolist()
              and all(len(set(r)) == K and 0 not in r and all(s in set(F.pool) for s in r) for r in rec.values()))
        checks[name] = ok
        log_print(fh, f'  wrote {path} ({sel[name]}): rows {len(out)}, format and pool check {"OK" if ok else "FAILED"}')
    stats['submission_checks'] = checks

    # ---------------- 4. two example users (same users for every method; chosen on V1 with the last-10-weeks history)
    spl, _ = keep[('V1', 'last10')]
    hist_n = spl.H[spl.H.user_id.isin(set(U))].groupby('user_id').size()
    cand = hist_n[[u in spl.truth for u in hist_n.index]]
    heavy = int(cand.idxmax()); light = int(cand[cand >= 3].idxmin())
    meta = sched[sched.tv_show_id != 0].drop_duplicates('tv_show_id', keep='last').set_index('tv_show_id')
    rows = []
    for role, u in (('heavy', heavy), ('light', light)):
        for fam in families:
            if fam == 'repeat_count':
                continue
            vname = sel[fam]; cfg = reg[vname][1]
            sp1, l1 = keep[('V1', cfg)]; F = finals[cfg]
            for window, S_, ids, truth in (('V1 26-30', sp1, sp1.finalize(l1[vname], K)[u], sp1.truth.get(u, set())),
                                           ('test 31-42', F, F.finalize(fl[fam], K)[u], None)):
                for r, s in enumerate(ids, 1):
                    rows.append({'user_role': role, 'user_id': str(u), 'window': window, 'method': fam, 'setting': vname,
                                 'rank': r, 'tv_show_id': s, 'title': meta.tv_show_title.get(s, ''),
                                 'category': meta.tv_show_category.get(s, ''),
                                 'watched_in_window_ge80': (s in truth) if truth is not None else '',
                                 'times_watched_in_history_ge50': int(((S_.H.user_id == u) & (S_.H.show == s)).sum())})
        hist = spl.H[spl.H.user_id == u].groupby('show').size().sort_values(ascending=False).head(10)
        stats[f'example_{role}'] = {'user_id': str(u), 'history_airings_ge50_weeks16_25': int(hist_n[u]),
                                    'distinct_programmes_history_weeks16_25': int(spl.H[spl.H.user_id == u].show.nunique()),
                                    'relevant_in_26_30': len(spl.truth.get(u, set())),
                                    'top_history_weeks16_25': [[int(s), meta.tv_show_title.get(s, ''), meta.tv_show_category.get(s, ''), int(c)] for s, c in hist.items()]}
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, 'examples_two_users.csv'), index=False)
    stats['selected'] = sel
    stats['timings_s'] = timings
    stats['total_runtime_s'] = round(time.time() - T0, 1)
    stats['environment'] = env
    json.dump(stats, open(os.path.join(res_dir, 'summary.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    log_print(fh, f'done in {stats["total_runtime_s"]} s; results in {res_dir}, submissions in {sub_dir}')


if __name__ == '__main__':
    main()
