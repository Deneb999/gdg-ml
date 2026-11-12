import numpy as np
import numba as nb

PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

K = 8
N_STATS = 4 + 4 + 2 + len(PAIRS) * (2 * K + 1)

@nb.njit(cache=True)
def prep(x, sr, xn, st, k):
    C, L = x.shape
    ln = float(L)
    nan = 0
    mu = np.empty(C, np.float64)
    s = np.empty(C, np.float64)

    for c in range(C):

        for t in range(L):
            v = x[c, t]
            if np.isnan(v):
                xn[c, t] = np.nan
                nan += 1
            else:
                xn[c, t] = v
        first = -1
        for t in range(L):
            if not np.isnan(xn[c, t]):
                first = t
                break
        if first < 0:
            for t in range(L):
                xn[c, t] = 0.0
            mu[c] = 0.0
            s[c] = 1e-6
            continue
        for t in range(first):
            xn[c, t] = xn[c, first]
        prev = first
        t = first + 1
        while t < L:
            if np.isnan(xn[c, t]):
                nxt = t + 1
                while nxt < L and np.isnan(xn[c, nxt]):
                    nxt += 1
                if nxt == L:
                    v0 = xn[c, prev]
                    for u in range(t, L):
                        xn[c, u] = v0
                    break
                v0 = xn[c, prev]
                v1 = xn[c, nxt]
                inv = 1.0 / (nxt - prev)
                for u in range(t, nxt):
                    xn[c, u] = v0 + (v1 - v0) * (u - prev) * inv
                prev = nxt
                t = nxt + 1
            else:
                prev = t
                t += 1

        acc = 0.0
        for t in range(L):
            acc += xn[c, t]
        m = acc / ln
        mu[c] = m
        v = 0.0
        for t in range(L):
            d = xn[c, t] - m
            v += d * d
        s[c] = np.sqrt(v / (ln - 1.0)) + 1e-6

    g = 0.0
    for c in range(C):
        g += s[c]
    g = g / C + 1e-6

    for c in range(C):
        st[c] = np.log(s[c] / g)
    st[8] = nan / (C * ln)
    st[9] = sr

    den = np.empty(C, np.float32)
    for c in range(C):
        mc = mu[c]
        for t in range(L):
            xn[c, t] = (xn[c, t] - mc) / g
        den[c] = np.dot(xn[c], xn[c])

    for c in range(C):
        st[4 + c] = np.dot(xn[c, : L - 1], xn[c, 1:]) / (ln - 1.0)

    base = 10
    for p in range(len(PAIRS)):
        ia = PAIRS[p][0]
        ib = PAIRS[p][1]
        dd = np.sqrt(den[ia] * den[ib]) + 1e-9
        for idx in range(2 * k + 1):
            l = idx - k
            if l >= 0:
                c = np.dot(xn[ia, : L - l], xn[ib, l:])
            else:
                c = np.dot(xn[ia, -l:], xn[ib, : L + l])
            st[base + p * (2 * k + 1) + idx] = c / dd
    return nan / (C * ln)

@nb.njit(cache=True)
def prep_fast(x, sr, xn, st):
    prep(x, sr, xn, st, K)
