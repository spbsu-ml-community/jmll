package com.spbsu.bernulli.MCMCBernoulliMixture;

public class UniformPrior implements BernoulliPrior {
  private final double[] logFactorialSums;  //logFactorialSums[i] = log(i!)

  public UniformPrior(int maxSize) {
    this.logFactorialSums = new double[maxSize];
    this.logFactorialSums[0] = 0;
    for (int i = 1; i < logFactorialSums.length; ++i) {
      this.logFactorialSums[i] = this.logFactorialSums[i-1]+Math.log(i);
    }
  }

  @Override
  public final double likelihood(int m, int n) {
    return logFactorialSums[m] + logFactorialSums[n - m] - logFactorialSums[n];
  }
}
