"""Recommenders. Every method scores the same candidate pool (labelled programmes that air in the
target window, repeats of already-watched programmes allowed) and returns, per user, a ranked list.
Lists shorter than K are filled from the popularity list (same fallback for every method)."""
import numpy as np, pandas as pd, scipy.sparse as sp

HOUR_SLOTS = 168  # hour-of-week


class Split:
    """One temporal split: history weeks (the last 10 training weeks) -> target weeks."""

    def __init__(self, A, log, sched, hist_weeks, target_weeks, users, hist_th=0.5, label_th=0.8):
        self.hw, self.tw = list(hist_weeks), list(target_weeks)
        self.last = max(self.hw)
        self.Aall = A[A.wk.isin(self.hw)]                       # every labelled overlap in the history window
        self.H = self.Aall[self.Aall.fmax >= hist_th]           # history interactions ("watched" = >= 50% in one sitting)
        self.log, self.sched = log, sched
        self.ts = sched[(sched.tv_show_id != 0) & sched.wk.isin(self.tw)]     # target-window airings (schedule only)
        self.pool = np.array(sorted(set(self.ts.tv_show_id)))
        self.items = np.array(sorted(set(self.H.show) | set(self.pool)))
        self.ii = {s: j for j, s in enumerate(self.items)}
        self.cmask = np.isin(self.items, self.pool)
        self.all_users = np.array(sorted(self.H.user_id.unique()), dtype=np.uint64)
        self.ui = {u: i for i, u in enumerate(self.all_users)}
        self.users = [u for u in users]
        self.eu = [u for u in users if u in self.ui]              # evaluated users with usable history
        self.erows = np.array([self.ui[u] for u in self.eu], dtype=int)
        cnt = self.H.groupby(['user_id', 'show']).size()
        r = np.array([self.ui[u] for u in cnt.index.get_level_values(0)])
        c = np.array([self.ii[s] for s in cnt.index.get_level_values(1)])
        self.X = sp.csr_matrix((cnt.values.astype(np.float64), (r, c)), shape=(len(self.all_users), len(self.items)))
        # popularity: distinct users who watched (>= 50%) in the last 4 history weeks, restricted to the pool
        h4 = self.H[(self.H.wk > self.last - 4) & self.H.show.isin(self.pool)]
        pop = h4.groupby('show').user_id.nunique()
        self.pop = pop.reindex(self.pool).fillna(0).sort_values(ascending=False, kind='stable')
        self.fallback = list(self.pop.index[:100])
        self.labels = None
        self.truth = {}
        if A.wk.max() >= min(self.tw):
            T = A[A.wk.isin(self.tw) & (A.fmax >= label_th)]
            self.truth = T[T.user_id.isin(set(users))].groupby('user_id').show.apply(set).to_dict()
            self.labels = T

    # ---- helpers
    def lists_from_dense(self, S, k=100):
        """Top-k pool items per evaluated user from a dense (len(eu) x len(items)) score matrix."""
        out = {}
        S = np.where(self.cmask[None, :], S, -np.inf)
        for row, u in enumerate(self.eu):
            v = S[row]
            top = np.argpartition(-v, min(k, len(v) - 1))[:k]
            top = top[np.isfinite(v[top]) & (v[top] > 0)]
            order = np.lexsort((self.items[top], -v[top]))       # score desc, then programme id asc
            out[u] = [(int(self.items[t]), float(v[t])) for t in top[order]]
        return out

    def finalize(self, lists, k=5):
        """Exactly k ids per user: method list, then popularity fallback, no duplicates."""
        res = {}
        for u in self.users:
            ids = [s for s, _ in lists.get(u, [])][:k]
            for p in self.fallback:
                if len(ids) >= k:
                    break
                if p not in ids:
                    ids.append(int(p))
            res[u] = ids
        return res


# ---------------- baselines
def popularity(sp_):
    top = [(int(s), float(v)) for s, v in sp_.pop.items()][:100]
    return {u: top for u in sp_.users}


def repeat_count(sp_):
    """Reference only (not one of the four methods): times the user watched the programme in the history window."""
    S = sp_.X[sp_.erows].toarray()
    return sp_.lists_from_dense(S)


# ---------------- association rules (item -> item, one transaction = the user's set of watched programmes)
def association_rules(sp_, min_support=5, self_rule=True):
    B = (sp_.X > 0).astype(np.float32)
    co = (B.T @ B).toarray()
    supp = np.diag(co).copy()
    np.fill_diagonal(co, 0)
    co[co < min_support] = 0
    conf = co / np.maximum(supp[:, None], 1)            # confidence(a -> b) = supp(a, b) / supp(a)
    if self_rule:
        np.fill_diagonal(conf, 1.0)                     # a -> a: a programme the user watches tends to recur
    S = np.asarray(sp_.X[sp_.erows] @ conf)             # sum over the user's programmes, weighted by watch count
    return sp_.lists_from_dense(S), conf


# ---------------- personalised PageRank on the programme co-view graph
def ppr(sp_, alpha=0.8, iters=40):
    B = (sp_.X > 0).astype(np.float64)
    G = (B.T @ B).tocsr()
    G.setdiag(0); G.eliminate_zeros()                   # edge weight = number of users who watched both
    deg = np.asarray(G.sum(1)).ravel()
    P = (sp.diags(1 / np.maximum(deg, 1e-12)) @ G).T.tocsr()
    dang = deg == 0
    R = sp_.X[sp_.erows].toarray()
    R = R / np.maximum(R.sum(1, keepdims=True), 1e-12)  # restart = the user's own watch distribution
    p = R.copy()
    for _ in range(iters):
        p = alpha * (P @ p.T).T + alpha * p[:, dang].sum(1, keepdims=True) * R + (1 - alpha) * R
    return sp_.lists_from_dense(p), p


# ---------------- implicit-feedback matrix factorisation (ALS, Hu-Koren-Volinsky)
def als(sp_, factors=16, reg=1.0, alpha=1.0, iters=15, seed=42):
    X = (sp_.X > 0).astype(np.float64).tocsr()
    rng = np.random.default_rng(seed)
    Uf = rng.normal(0, 0.01, (X.shape[0], factors)); If = rng.normal(0, 0.01, (X.shape[1], factors))
    Xt = X.T.tocsr()

    def solve(Y, M):
        YtY = Y.T @ Y
        out = np.zeros((M.shape[0], factors))
        for r in range(M.shape[0]):
            s, e = M.indptr[r], M.indptr[r + 1]
            if s == e:
                continue
            Yi = Y[M.indices[s:e]]; c = 1 + alpha * M.data[s:e]
            out[r] = np.linalg.solve(YtY + (Yi.T * (c - 1)) @ Yi + reg * np.eye(factors), (Yi.T * c).sum(1))
        return out

    for _ in range(iters):
        Uf = solve(If, X); If = solve(Uf, Xt)
    S = Uf[sp_.erows] @ If.T
    return sp_.lists_from_dense(S), S


# ---------------- content-based filtering
def _tfidf_rows(M):
    df = np.asarray((M > 0).sum(0)).ravel()
    M = M @ sp.diags(np.log((1 + M.shape[0]) / (1 + df)) + 1)
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n) @ M


def cbf_metadata(sp_):
    """Item vector: TF-IDF over category, genres, director, actors, decade of production."""
    meta = sp_.sched[sp_.sched.tv_show_id != 0].drop_duplicates('tv_show_id', keep='last').set_index('tv_show_id')
    vocab, rows, cols = {}, [], []
    for i, s in enumerate(sp_.items):
        if s not in meta.index:
            continue
        r = meta.loc[s]
        toks = {'cat=' + r.tv_show_category}
        toks |= {'g=' + g for g in (r.tv_show_genre_1, r.tv_show_genre_2, r.tv_show_genre_3) if g}
        toks |= {'d=' + x.strip() for x in r.director.split(',') if x.strip()}
        toks |= {'a=' + x.strip() for x in r.actors.split(',') if x.strip()}
        if r.year_of_production[:4].isdigit():
            toks.add('dec=' + r.year_of_production[:3])
        for t in toks:
            rows.append(i); cols.append(vocab.setdefault(t, len(vocab)))
    M = _tfidf_rows(sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sp_.items), len(vocab))))
    prof = sp_.X[sp_.erows] @ M
    pn = np.sqrt(np.asarray(prof.multiply(prof).sum(1)).ravel()); pn[pn == 0] = 1
    S = np.asarray((sp.diags(1 / pn) @ prof @ M.T).todense())
    return sp_.lists_from_dense(S), S


def hour_of_week(ts_values):
    h = ts_values.astype('datetime64[s]').astype(np.int64) // 3600
    return ((h // 24 + 3) % 7) * 24 + h % 24          # 1970-01-01 was a Thursday; Monday 00h = slot 0


def view_hours(log, weeks, users):
    """Split each view of the given users into hour-of-week pieces: (user, channel, slot, seconds)."""
    v = log[log.wk.isin(weeks) & log.user_id.isin(set(users))]
    s = v.start.values.astype('datetime64[s]').astype(np.int64); e = v.stop.values.astype('datetime64[s]').astype(np.int64)
    h0, h1 = s // 3600, (e - 1) // 3600
    n = (h1 - h0 + 1).clip(1, HOUR_SLOTS)
    rep = np.repeat(np.arange(len(v)), n)
    hh = h0[rep] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    sec = np.minimum(e[rep], (hh + 1) * 3600) - np.maximum(s[rep], hh * 3600)
    df = pd.DataFrame({'user_id': v.user_id.values[rep], 'ch': v.vsetv_id.values[rep],
                       'slot': ((hh // 24 + 3) % 7) * 24 + hh % 24, 'sec': sec})
    return df[df.sec > 0]


def cbf_schedule(sp_):
    """Content-based filtering on schedule attributes: a programme is described by when and where it airs
    in the target window (channel x hour-of-week counts per week); the user profile is their viewing
    hours per channel x hour-of-week in the history window, unlabelled airtime included.
    Score = profile . item vector = expected hours per week the user is already watching that channel
    while the programme is on air. Uses the target-window schedule only (never target-window views)."""
    V = view_hours(sp_.log, sp_.hw, sp_.eu)
    chans = sorted(set(V.ch) | set(sp_.ts.channel_id)); cix = {c: i for i, c in enumerate(chans)}
    uix = {u: i for i, u in enumerate(sp_.eu)}
    prof = sp.csr_matrix((V.sec.values / 3600.0 / len(sp_.hw), (V.user_id.map(uix).values, V.ch.map(cix).values * HOUR_SLOTS + V.slot.values)),
                         shape=(len(sp_.eu), len(chans) * HOUR_SLOTS))
    ts = sp_.ts
    key = ts.channel_id.map(cix).values * HOUR_SLOTS + hour_of_week(ts.start.values)
    item = sp.csr_matrix((np.ones(len(ts)) / len(sp_.tw), (ts.tv_show_id.map(sp_.ii).values, key)),
                         shape=(len(sp_.items), len(chans) * HOUR_SLOTS))
    S = (prof @ item.T).toarray()
    return sp_.lists_from_dense(S), S, V


# ---------------- hybrid: weighted reciprocal-rank fusion
def rrf(list_a, list_b, w_a=0.7, k=60):
    out = {}
    for u in set(list_a) | set(list_b):
        sc = {}
        for lst, w in ((list_a.get(u, []), w_a), (list_b.get(u, []), 1 - w_a)):
            for r, (s, _) in enumerate(lst):
                sc[s] = sc.get(s, 0.0) + w / (k + r + 1)
        out[u] = sorted(sc.items(), key=lambda x: (-x[1], x[0]))[:100]
    return out
