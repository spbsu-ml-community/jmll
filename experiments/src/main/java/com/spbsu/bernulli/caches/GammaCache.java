package com.spbsu.bernulli.caches;

import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

/**
 * Created by noxoomo on 11/04/15.
 */
public class GammaCache {
  final double alpha;
  final double[] logGamma;
  private int current;

  public GammaCache(double alpha, int maxSize) {
    this.alpha = alpha;
    this.logGamma = new double[maxSize+1];
    this.current = 0;
    this.logGamma[0] = Gamma.logGamma(alpha);
  }

  public double logValue(int n) {
    assert(n < logGamma.length);
    if (current < n) {
      for (int i=current+1; i < n+1;++i) {
        logGamma[i] = logGamma[i-1] + FastMath.log(i + alpha - 1);
      }
      current = n;
    }
    return logGamma[n];
  }

  public double value(int n) {
    return FastMath.exp(logValue(n));
  }
}
