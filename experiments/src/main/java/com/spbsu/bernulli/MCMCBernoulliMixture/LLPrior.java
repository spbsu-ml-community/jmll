package com.spbsu.bernulli.MCMCBernoulliMixture;

import org.apache.commons.math3.util.FastMath;

public class LLPrior implements BernoulliPrior {

  @Override
  public final double likelihood(int m, int n) {
    final double p = m  * 1.0 / n;
    return m * FastMath.log(p) + (n-m) * FastMath.log(1-p);
  }
}
