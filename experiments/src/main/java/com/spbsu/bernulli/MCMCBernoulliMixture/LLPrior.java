package com.spbsu.bernulli.MCMCBernoulliMixture;

public class LLPrior implements BernoulliPrior {

  public LLPrior() {
  }

  @Override
  public final double likelihood(int m, int n) {
    final double p = m  * 1.0 / n;
    return m * Math.log(p) + (n-m) * Math.log(1-p);
  }
}
