package com.spbsu.bernulli.caches;

public class BetaCache {
  double alpha;
  double beta;
  int n;

  public BetaCache(double alpha, double beta, int n) {
    this.alpha = alpha;
    this.beta = beta;
    this.n = n;
    logAlpha = new double[n + 1];
    logBeta = new double[n + 1];
    logAlphaBetaSum = 0;
  }

  final double logAlpha[];
  int lastCachedAlpha = 0;
  final double logBeta[];
  int lastCachedBeta = 0;
  double logAlphaBetaSum;
  boolean cachedAlphaBetaSum = false;


  final public double calculate(int m, int n) {
    if (!cachedAlphaBetaSum) {
      final double ab = alpha + beta;
      for (int i = 0; i < n; ++i) {
        logAlphaBetaSum += Math.log(ab + i);
      }
      cachedAlphaBetaSum = true;
    }
    if (lastCachedAlpha < m) {
      for (int i = lastCachedAlpha + 1; i <= m; ++i) {
        logAlpha[i] = logAlpha[i - 1] + Math.log(alpha + i - 1);
      }
      lastCachedAlpha = m;
    }
    if (lastCachedBeta < n - m) {
      for (int i = lastCachedBeta + 1; i <= n - m; ++i) {
        logBeta[i] = logBeta[i - 1] + Math.log(beta + i - 1);
      }
      lastCachedBeta = n - m;
    }
    return Math.exp(logAlpha[m] + logBeta[n - m] - logAlphaBetaSum);
  }

  final public void update(double alpha, double beta) {
    if (this.alpha == alpha && this.beta == beta)
      return;
    this.alpha = alpha;
    this.beta = beta;
    logAlpha[0] = 0;
    logBeta[0] = 0;
    lastCachedAlpha = 0;
    lastCachedBeta = 0;
    logAlphaBetaSum = 0;
    cachedAlphaBetaSum = false;
  }

}
