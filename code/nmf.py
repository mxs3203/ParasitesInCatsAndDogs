"""Bayesian Bernoulli matrix factorisation (Lumbreras et al. 2020) for presence/absence results with missing entries.

    x_ij ~ Bernoulli(sum_k w_ik h_kj),  h_kj ~ Beta(a, b),  w_i ~ Dirichlet(alpha_g(i))

Fitted by collapsed Gibbs sampling with a move that reassigns all results of a sample at once; each alpha is updated
by a random-walk Metropolis step on the log scale (Gamma(1, 1) prior). Samples can share one prior (g = 0) or have
one per region. The module also holds the Bernoulli mixture used to start the chains and the two-step multinomial
model (Bakk & Kuha 2018) that relates profile membership to covariates.
"""
import math

import numpy as np
from numba import njit
from scipy.optimize import minimize

EPS = 1e-10


def unique_rows(X, M):
    """Collapse identical (result, observed) rows: unique X, unique M, counts and the inverse index."""
    key = np.hstack([X, M]).astype(np.int8)
    u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    J = X.shape[1]
    return u[:, :J].astype(float), u[:, J:].astype(float), cnt.astype(float), inv.ravel()


def cosine(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


# ------------------------------------------------------------------ Bernoulli mixture (starting values)
def _class_loglik(X, M, H):
    lh, l1h = np.log(np.clip(H, EPS, 1 - EPS)), np.log(np.clip(1 - H, EPS, 1 - EPS))
    return (M * X) @ lh.T + (M * (1 - X)) @ l1h.T


def _posterior(X, M, pi, H):
    lc = _class_loglik(X, M, H) + np.log(pi)
    mx = lc.max(1, keepdims=True)
    lse = mx + np.log(np.exp(lc - mx).sum(1, keepdims=True))
    return np.exp(lc - lse), lse.ravel()


def mixture_fit(X, M, K, c=None, n_init=50, max_iter=5000, tol=1e-10, seed=0):
    """Mixture of independent Bernoulli distributions by EM, best of n_init random starts."""
    rng = np.random.default_rng(seed)
    N, J = X.shape
    c = np.ones(N) if c is None else c
    best = None
    for _ in range(n_init):
        pi = rng.dirichlet(np.ones(K))
        H = rng.uniform(0.02, 0.6, size=(K, J))
        prev = -np.inf
        for _ in range(max_iter):
            post, lse = _posterior(X, M, pi, H)
            ll = np.sum(c * lse)
            pc = post * c[:, None]
            pi = np.clip(pc.sum(0) / c.sum(), EPS, None)
            pi /= pi.sum()
            H = np.clip((pc.T @ (M * X)) / np.clip(pc.T @ M, EPS, None), EPS, 1 - EPS)
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        ll = np.sum(c * _posterior(X, M, pi, H)[1])
        if best is None or ll > best["loglik"]:
            best = dict(pi=pi, H=H, loglik=ll)
    order = np.argsort(-best["pi"])
    best["pi"], best["H"] = best["pi"][order], best["H"][order]
    return best


def start(X, M, K, n_init, seed):
    """Profile of each sample under the Bernoulli mixture, used to initialise a chain."""
    xu, mu, c, inv = unique_rows(X, M)
    m = mixture_fit(xu, mu, K, c=c, n_init=n_init, seed=seed)
    return _posterior(xu, mu, m["pi"], m["H"])[0].argmax(1)[inv]


# ------------------------------------------------------------------ Gibbs sampler
@njit(cache=True)
def _log_pz_group(nik, ni, grp, alpha, g):
    N, K = nik.shape
    A = 0.0
    for k in range(K):
        A += alpha[g, k]
    s = 0.0
    lgA = math.lgamma(A)
    for i in range(N):
        if grp[i] != g:
            continue
        s += lgA - math.lgamma(ni[i] + A)
        for k in range(K):
            s += math.lgamma(nik[i, k] + alpha[g, k]) - math.lgamma(alpha[g, k])
    return s


@njit(cache=True)
def _log_px_z(s1, s0, a, b):
    s = 0.0
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    for k in range(s1.shape[0]):
        for j in range(s1.shape[1]):
            s += math.lgamma(s1[k, j] + a) + math.lgamma(s0[k, j] + b) - math.lgamma(s1[k, j] + s0[k, j] + a + b) - lb
    return s


@njit(cache=True)
def _gibbs(tr, tc, tx, row_start, grp, G, N, J, K, alpha0, a, b, n_sweeps, burn, thin, seed, step, z_init):
    np.random.seed(seed)
    T = tr.shape[0]
    alpha = np.full((G, K), alpha0)
    z = np.empty(T, np.int64)
    nik = np.zeros((N, K)); ni = np.zeros(N); s1 = np.zeros((K, J)); s0 = np.zeros((K, J))
    for t in range(T):
        k = z_init[t] if z_init[0] >= 0 else np.random.randint(K)
        z[t] = k; nik[tr[t], k] += 1; ni[tr[t]] += 1
        if tx[t] == 1:
            s1[k, tc[t]] += 1
        else:
            s0[k, tc[t]] += 1
    H_sum = np.zeros((K, J)); alpha_sum = np.zeros((G, K)); ll_tr = np.zeros(n_sweeps)
    p = np.zeros(K); n_keep = 0
    for sweep in range(n_sweeps):
        # one result at a time
        for t in range(T):
            i = tr[t]; j = tc[t]; x = tx[t]; k = z[t]; gi = grp[i]
            nik[i, k] -= 1
            if x == 1:
                s1[k, j] -= 1
            else:
                s0[k, j] -= 1
            tot = 0.0
            for kk in range(K):
                den = s1[kk, j] + s0[kk, j] + a + b
                lik = (s1[kk, j] + a) / den if x == 1 else (s0[kk, j] + b) / den
                tot += (nik[i, kk] + alpha[gi, kk]) * lik
                p[kk] = tot
            u = np.random.random() * tot
            k = 0
            while p[k] < u:
                k += 1
            z[t] = k; nik[i, k] += 1
            if x == 1:
                s1[k, j] += 1
            else:
                s0[k, j] += 1
        # all results of a sample together, for samples whose results sit in a single profile
        for i in range(N):
            if ni[i] == 0:
                continue
            kc = -1
            for kk in range(K):
                if nik[i, kk] == ni[i]:
                    kc = kk
                    break
            if kc < 0:
                continue
            gi = grp[i]; t0 = row_start[i]; t1 = row_start[i + 1]
            for t in range(t0, t1):
                if tx[t] == 1:
                    s1[kc, tc[t]] -= 1
                else:
                    s0[kc, tc[t]] -= 1
            nik[i, kc] = 0
            mxl = -1e300
            for kk in range(K):
                lw = math.lgamma(ni[i] + alpha[gi, kk]) - math.lgamma(alpha[gi, kk])
                for t in range(t0, t1):
                    jj = tc[t]; den = s1[kk, jj] + s0[kk, jj] + a + b
                    lw += math.log((s1[kk, jj] + a) / den) if tx[t] == 1 else math.log((s0[kk, jj] + b) / den)
                p[kk] = lw
                if lw > mxl:
                    mxl = lw
            tot = 0.0
            for kk in range(K):
                tot += math.exp(p[kk] - mxl); p[kk] = tot
            u = np.random.random() * tot
            kn = 0
            while p[kn] < u:
                kn += 1
            for t in range(t0, t1):
                z[t] = kn
                if tx[t] == 1:
                    s1[kn, tc[t]] += 1
                else:
                    s0[kn, tc[t]] += 1
            nik[i, kn] = ni[i]
        # Dirichlet parameters
        lpz = 0.0
        for g in range(G):
            cur = _log_pz_group(nik, ni, grp, alpha, g)
            for kk in range(K):
                old = alpha[g, kk]
                prop = math.exp(math.log(old) + step * np.random.standard_normal())
                if prop < 1e-6:
                    continue
                alpha[g, kk] = prop
                new = _log_pz_group(nik, ni, grp, alpha, g)
                if math.log(np.random.random()) < (new - prop + math.log(prop)) - (cur - old + math.log(old)):
                    cur = new
                else:
                    alpha[g, kk] = old
            lpz += cur
        ll_tr[sweep] = _log_px_z(s1, s0, a, b) + lpz
        if sweep >= burn and (sweep - burn) % thin == 0:
            for kk in range(K):
                for jj in range(J):
                    H_sum[kk, jj] += (s1[kk, jj] + a) / (s1[kk, jj] + s0[kk, jj] + a + b)
                for g in range(G):
                    alpha_sum[g, kk] += alpha[g, kk]
            n_keep += 1
    return H_sum / n_keep, alpha_sum / n_keep, ll_tr


def fit(X, M, grp, K, row_init=None, alpha0=0.3, a=1.0, b=1.0, n_sweeps=2500, burn=1000, thin=5, seed=0, step=0.2):
    """Posterior means of H (K x J) and alpha (groups x K), and the mean log joint probability after burn-in."""
    r, c = np.nonzero(M)
    tr, tc, tx = r.astype(np.int64), c.astype(np.int64), X[r, c].astype(np.int64)
    N, J = X.shape
    row_start = np.zeros(N + 1, np.int64)
    row_start[1:] = np.cumsum(np.bincount(tr, minlength=N))
    grp = np.asarray(grp, np.int64)
    z_init = np.full(tr.shape[0], -1, np.int64) if row_init is None else np.asarray(row_init, np.int64)[tr]
    H, alpha, trace = _gibbs(tr, tc, tx, row_start, grp, int(grp.max()) + 1, N, J, K, float(alpha0), float(a), float(b),
                             n_sweeps, burn, thin, seed, step, z_init)
    return dict(H=H, alpha=alpha, loglik_mean=float(trace[burn:].mean()))


def sample_dirichlet_log(alpha, S, rng):
    """Dirichlet draws computed on the log scale, stable for alpha close to 0."""
    g = np.log(rng.gamma(alpha + 1.0, size=(S, alpha.size))) + np.log(rng.uniform(size=(S, alpha.size))) / alpha
    g -= g.max(1, keepdims=True)
    w = np.exp(g)
    return w / w.sum(1, keepdims=True)


def predictive_loo(X, M, grp, H, alpha, S=50_000, seed=0, chunk=250):
    """P(x_ij = 1 | the other observed results of sample i), by importance sampling from the Dirichlet prior."""
    rng = np.random.default_rng(seed)
    N, J = X.shape
    P = np.full((N, J), np.nan)
    for g in np.unique(grp):
        rows = np.where(grp == g)[0]
        Ws = sample_dirichlet_log(alpha[g], S, rng)
        V = np.clip(Ws @ H, 1e-12, 1 - 1e-12)
        lV, l1V = np.log(V), np.log(1 - V)
        for c0 in range(0, len(rows), chunk):
            rr = rows[c0:c0 + chunk]
            Xg, Mg = X[rr], M[rr]
            for j in range(J):
                Mj = Mg.copy(); Mj[:, j] = 0
                lw = (Mj * Xg) @ lV.T + (Mj * (1 - Xg)) @ l1V.T
                lw -= lw.max(1, keepdims=True)
                w = np.exp(lw)
                P[rr, j] = (w @ V[:, j]) / w.sum(1)
    return np.clip(P, 1e-12, 1 - 1e-12)


def weights(X, M, H, alpha, S=50_000, seed=0, chunk=200):
    """Posterior mean profile weights E[w_i | x_i], by importance sampling from the Dirichlet prior."""
    rng = np.random.default_rng(seed)
    Ws = sample_dirichlet_log(alpha, S, rng)
    V = np.clip(Ws @ H, 1e-12, 1 - 1e-12)
    lV, l1V = np.log(V), np.log(1 - V)
    xu, mu, _, inv = unique_rows(X, M)
    out = np.zeros((len(xu), H.shape[0]))
    for c0 in range(0, len(xu), chunk):
        xs, ms = xu[c0:c0 + chunk], mu[c0:c0 + chunk]
        lw = (ms * xs) @ lV.T + (ms * (1 - xs)) @ l1V.T
        lw -= lw.max(1, keepdims=True)
        w = np.exp(lw)
        out[c0:c0 + chunk] = (w @ Ws) / w.sum(1, keepdims=True)
    return out[inv]


def loglik(P, X, M):
    return float(np.sum(M * (X * np.log(P) + (1 - X) * np.log(1 - P))))


# ------------------------------------------------------------------ two-step model of profile membership
def aggregate(X, M, Z):
    """Collapse identical (results, observed, covariates) rows."""
    key = np.hstack([X, M, Z])
    u, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    J, P = X.shape[1], Z.shape[1]
    return u[:, :J], u[:, J:2 * J], u[:, 2 * J:2 * J + P], cnt.astype(float), inv.ravel()


def _softmax(A):
    A = A - A.max(1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(1, keepdims=True)


def twostep_fit(X, M, Z, H, c=None, max_iter=2000, tol=1e-9, ridge=1e-4):
    """Multinomial logit of profile membership on Z (with an intercept column), profiles H held fixed.
    EM with an L-BFGS M-step; returns the coefficients B (P x K, first profile as reference) and the log-likelihood."""
    N = X.shape[0]
    P, K = Z.shape[1], H.shape[0]
    c = np.ones(N) if c is None else c
    lh, l1h = np.log(np.clip(H, EPS, 1 - EPS)), np.log(np.clip(1 - H, EPS, 1 - EPS))
    lc = (M * X) @ lh.T + (M * (1 - X)) @ l1h.T
    B = np.zeros((P, K))
    prev = -np.inf
    for _ in range(max_iter):
        a = lc + np.log(np.clip(_softmax(Z @ B), EPS, 1))
        mx = a.max(1, keepdims=True)
        lse = mx + np.log(np.exp(a - mx).sum(1, keepdims=True))
        qc = np.exp(a - lse) * c[:, None]
        ll = float(np.sum(c * lse.ravel()))

        def objective(b):
            Bm = np.zeros((P, K))
            Bm[:, 1:] = b.reshape(P, K - 1)
            A = Z @ Bm
            A = A - A.max(1, keepdims=True)
            lsm = A - np.log(np.exp(A).sum(1, keepdims=True))
            f = -np.sum(qc * lsm) + ridge * np.sum(b ** 2)
            g = -(Z.T @ (qc - c[:, None] * np.exp(lsm)))[:, 1:].ravel() + 2 * ridge * b
            return f, g

        B[:, 1:] = minimize(objective, B[:, 1:].ravel(), jac=True, method="L-BFGS-B", options=dict(maxiter=50)).x.reshape(P, K - 1)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return dict(B=B, loglik=ll)


def shares(Z, B, levels, w=None):
    """Marginally standardised profile shares: every row set in turn to each level of a categorical covariate.
    levels maps each level to its column in Z (None for the reference level)."""
    w = np.ones(Z.shape[0]) if w is None else w
    cols = [j for j in levels.values() if j is not None]
    out = {}
    for lev, j in levels.items():
        Zs = Z.copy()
        Zs[:, cols] = 0
        if j is not None:
            Zs[:, j] = 1
        out[lev] = (_softmax(Zs @ B) * w[:, None]).sum(0) / w.sum()
    return out
