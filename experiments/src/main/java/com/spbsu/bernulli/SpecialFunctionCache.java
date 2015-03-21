package com.spbsu.bernulli;

import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

//calculate and cache Beta(alpha + m, beta + n -m) / Beta(alpha,beta)
public class SpecialFunctionCache {

  class DigammaCache {
    int maxOffset;
    double base;
    final double[] values;
    final boolean[] cached;

    DigammaCache(double base, int n) {
      values = new double[n + 2];
      cached = new boolean[n + 2];
      this.base = base;
      this.maxOffset = n;
    }

    // limits for switching algorithm in digamma
    /**
     * C limit.
     */
    private static final double C_LIMIT = 49;

    /**
     * S limit.
     */
    private static final double S_LIMIT = 1e-5;

    final double calc(int offset) {
      if (cached[offset]) {
        return values[offset];
      } else {
        double x = base + offset;
        if (x > 0 && x <= S_LIMIT) {
          // use method 5 from Bernardo AS103
          // accurate to O(x)
          values[offset] = -Gamma.GAMMA - 1 / x;
          cached[offset] = true;
        } else if (x >= C_LIMIT) {
          // use method 4 (accurate to O(1/x^8)
          double inv = 1 / (x * x);
          //            1       1        1         1
          // log(x) -  --- - ------ + ------- - -------
          //           2 x   12 x^2   120 x^4   252 x^6
          values[offset] = FastMath.log(x) - 0.5 / x - inv * ((1.0 / 12) + inv * (1.0 / 120 - inv / 252));
          cached[offset] = true;
        } else if (offset == (n + 1)) {
          cached[offset] = true;
          values[offset] = Gamma.digamma(base + offset);
        } else {
          values[offset] = calc(offset + 1) - 1.0 / x;
          cached[offset] = true;
        }
      }
      return values[offset];
    }

    final public void update(double newBase) {
      this.base = newBase;
      Arrays.fill(cached, false);
    }
  }

  class Digamma1Cache {
    int maxOffset;
    double base;
    final double[] values;
    final boolean[] cached;

    Digamma1Cache(double base, int n) {
      values = new double[n + 2];
      cached = new boolean[n + 2];
      this.base = base;
      this.maxOffset = n;
    }

    // limits for switching algorithm in digamma
    /**
     * C limit.
     */
    private static final double C_LIMIT = 49;

    /**
     * S limit.
     */
    private static final double S_LIMIT = 1e-5;

    final double calc(int offset) {
      if (cached[offset]) {
        return values[offset];
      } else {
        double x = base + offset;
        if (x > 0 && x <= S_LIMIT) {
          // use method 5 from Bernardo AS103
          // accurate to O(x)
          values[offset] = 1.0 / (x * x);
          cached[offset] = true;
        } else if (x >= C_LIMIT) {
          final double inv = 1.0 / (x * x);
          //  1    1      1       1       1
          //  - + ---- + ---- - ----- + -----
          //  x      2      3       5       7
          //      2 x    6 x    30 x    42 x
          values[offset] = 1.0 / x + inv / 2 + inv / x * (1.0 / 6 - inv * (1.0 / 30 + inv / 42));
          cached[offset] = true;
        } else if (offset == (n + 1)) {
          cached[offset] = true;
          values[offset] = Gamma.trigamma(base + offset);
        } else {
          values[offset] = calc(offset + 1) + 1 / (x * x);
          cached[offset] = true;
        }
      }
      return values[offset];
    }


    final public void update(double newBase) {
      this.base = newBase;
      Arrays.fill(cached, false);
    }
  }


  double alpha;
  double beta;
  int n;
  DigammaCache[] digammaCaches = new DigammaCache[3];
  Digamma1Cache[] digamma1Caches = new Digamma1Cache[3];
  final double logAlpha[];
  int lastCachedAlpha = 0;
  final double logBeta[];
  int lastCachedBeta = 0;
  double logAlphaBetaSum;
  boolean cachedAlphaBetaSum = false;


  public SpecialFunctionCache(double alpha, double beta, int n) {
    this.alpha = alpha;
    this.beta = beta;
    this.n = n;
    digammaCaches[0] = new DigammaCache(alpha, n);
    digammaCaches[1] = new DigammaCache(beta, n);
    digammaCaches[2] = new DigammaCache(alpha + beta, n);

    digamma1Caches[0] = new Digamma1Cache(alpha, n);
    digamma1Caches[1] = new Digamma1Cache(beta, n);
    digamma1Caches[2] = new Digamma1Cache(alpha + beta, n);
    logAlpha = new double[n + 1];
    logBeta = new double[n + 1];
    logAlphaBetaSum = 0;
  }

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


  public enum Type {
    Alpha,
    Beta,
    AlphaBeta
  }

  final public double digamma(Type type, int offset) {
    if (type == Type.Alpha) {
      return digammaCaches[0].calc(offset);
    } else if (type == Type.Beta) {
      return digammaCaches[1].calc(offset);
    } else {
      return digammaCaches[2].calc(offset);
    }
  }

  public double digamma1(Type type, int offset) {
    if (type == Type.Alpha) {
      return digamma1Caches[0].calc(offset);
    } else if (type == Type.Beta) {
      return digamma1Caches[1].calc(offset);
    } else {
      return digamma1Caches[2].calc(offset);
    }
  }


  final public void update(double alpha, double beta) {
    this.alpha = alpha;
    this.beta = beta;
    logAlpha[0] = 0;
    logBeta[0] = 0;
    lastCachedAlpha = 0;
    lastCachedBeta = 0;
    logAlphaBetaSum = 0;
    cachedAlphaBetaSum = false;
    digammaCaches[0].update(alpha);
    digammaCaches[1].update(beta);
    digammaCaches[2].update(alpha + beta);

    digamma1Caches[0].update(alpha);
    digamma1Caches[1].update(beta);
    digamma1Caches[2].update(alpha + beta);
  }
}
