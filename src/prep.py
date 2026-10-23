"""Step 1: read the Kaggle files, clean the schedule and join views to programme airings.

Join rule: a view (user, vsetv_id, start_time, stop_time) is matched to every schedule row on the
same channel (vsetv_id == channel_id) whose airing interval [start, start + duration) overlaps the
view interval. Rows with tv_show_id == 0 (unlabelled airtime) are never items; they are kept only in
the raw view log, which the schedule-based content model uses as viewing context.
Per (user, airing) we keep the summed overlap (capped at the programme length) and the longest
single-view overlap. `fmax` = longest single-view overlap / programme duration ("single sitting").
"""
import os, time
import numpy as np, pandas as pd

LOG_FILE = 'dataset11-30.csv'
SCHED_FILES = ['export_arh_11-20-final.csv', 'export_arh_21-30-final.csv', 'export_arh_31-42-final.csv']
META_COLS = ['tv_show_title', 'tv_show_category', 'tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3',
             'year_of_production', 'director', 'actors']


def iso_week(ts):
    return ts.dt.isocalendar().week.astype(int).values


def read_log(data_dir):
    log = pd.read_csv(os.path.join(data_dir, LOG_FILE),
                      dtype={'user_id': 'uint64', 'vsetv_id': 'int32', 'duraton': 'int64'})
    log['start'] = pd.to_datetime(log.start_time)
    log['stop'] = pd.to_datetime(log.stop_time)
    log = log.drop(columns=['start_time', 'stop_time'])
    log['wk'] = iso_week(log.start)
    return log


def read_schedule(data_dir):
    parts = []
    for f in SCHED_FILES:
        s = pd.read_csv(os.path.join(data_dir, f), dtype=str, keep_default_na=False)
        s['source_file'] = f
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    raw_rows = len(s)
    s['start'] = pd.to_datetime(s.start_time, format='%d.%m.%Y %H:%M:%S')
    s['duration'] = s.duration.astype('int64')
    s['channel_id'] = s.channel_id.astype('int32')
    s['tv_show_id'] = s.tv_show_id.astype('int64')
    s = s.drop(columns=['start_time'])
    # neighbouring files overlap by a few hours: keep one row per (channel, start), preferring a labelled one
    s = s.sort_values(['channel_id', 'start', 'tv_show_id'], ascending=[True, True, False])
    s = s.drop_duplicates(['channel_id', 'start']).reset_index(drop=True)
    s['end'] = s.start + pd.to_timedelta(s.duration, unit='s')
    s['wk'] = iso_week(s.start)
    s['pid'] = np.arange(len(s))
    return s, raw_rows


def interval_join(log, sched):
    """All (view, schedule row) pairs on the same channel with positive overlap, in seconds."""
    out = []
    for ch, g in log.groupby('vsetv_id'):
        sc = sched[sched.channel_id == ch]
        if len(sc) == 0:
            continue
        st, en, pid = sc.start.values, sc.end.values, sc.pid.values
        vs, ve = g.start.values, g.stop.values
        i0 = np.maximum(np.searchsorted(st, vs, 'right') - 1, 0)
        i1 = np.searchsorted(st, ve, 'left') - 1
        n = np.maximum(i1 - i0 + 1, 0)
        if n.sum() == 0:
            continue
        rep = np.repeat(np.arange(len(g)), n)
        k = np.repeat(i0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        ov = (np.minimum(ve[rep], en[k]) - np.maximum(vs[rep], st[k])) / np.timedelta64(1, 's')
        m = ov > 0
        out.append(pd.DataFrame({'vid': g.index.values[rep[m]], 'pid': pid[k[m]], 'ov': ov[m]}))
    return pd.concat(out, ignore_index=True)


def build_airings(log, sched, pairs):
    p = pairs.merge(sched[['pid', 'tv_show_id']], on='pid')
    p = p[p.tv_show_id != 0]
    p['user_id'] = log.user_id.values[p.vid.values]
    a = p.groupby(['user_id', 'pid'], as_index=False).agg(ov=('ov', 'sum'), ovmax=('ov', 'max'))
    a = a.merge(sched[['pid', 'channel_id', 'start', 'duration', 'tv_show_id', 'wk']], on='pid')
    dur = a.duration.clip(lower=1)
    a['ov'] = np.minimum(a.ov, dur)
    a['fmax'] = a.ovmax / dur
    return a.rename(columns={'tv_show_id': 'show', 'channel_id': 'ch'})


def prepare(data_dir, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    paths = {k: os.path.join(work_dir, k + '.parquet') for k in ('log', 'sched', 'airings')}
    stats = {}
    t0 = time.time()
    if all(os.path.exists(p) for p in paths.values()):
        log, sched, A = (pd.read_parquet(paths[k]) for k in ('log', 'sched', 'airings'))
        stats['cache'] = 'reused parquet cache in ' + work_dir
    else:
        log = read_log(data_dir)
        sched, raw = read_schedule(data_dir)
        pairs = interval_join(log, sched)
        A = build_airings(log, sched, pairs)
        lab = pairs.merge(sched[['pid', 'tv_show_id']], on='pid')
        stats.update(schedule_rows_raw=raw, schedule_rows_dedup=len(sched), view_schedule_pairs=len(pairs),
                     view_seconds=int(log.duraton.sum()),
                     share_view_seconds_on_any_schedule_row=float(pairs.ov.sum() / log.duraton.sum()),
                     share_view_seconds_on_labelled_rows=float(lab.loc[lab.tv_show_id != 0, 'ov'].sum() / log.duraton.sum()))
        log.to_parquet(paths['log']); sched.to_parquet(paths['sched']); A.to_parquet(paths['airings'])
    stats.update(log_rows=len(log), log_users=int(log.user_id.nunique()), log_channels=int(log.vsetv_id.nunique()),
                 log_first=str(log.start.min()), log_last=str(log.stop.max()),
                 labelled_schedule_rows=int((sched.tv_show_id != 0).sum()),
                 labelled_share_by_file={f: float((g.tv_show_id != 0).mean()) for f, g in sched.groupby('source_file')},
                 user_airing_pairs_any_overlap=len(A), prep_seconds=round(time.time() - t0, 1))
    return log, sched, A, stats
