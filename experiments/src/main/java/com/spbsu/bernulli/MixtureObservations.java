package com.spbsu.bernulli;

import static com.spbsu.commons.math.MathTools.sqr;

public class MixtureObservations<MixtureDistribution> {
  public final MixtureDistribution owner;
  public final int components[];
  public final double[] thetas; //theta[i] ~ Beta(alpha[components[i]], beta[components[i]])
  public final int sums[]; //sums[i] = Sum Bernoulli(theta[i]
  public final int n;

  public MixtureObservations(MixtureDistribution owner, int[] components, double[] thetas, int[] sums, int n) {
    this.owner = owner;
    this.components = components;
    this.thetas = thetas;
    this.sums = sums;
    this.n = n;
  }

  public final double quality(final double[] estimator) {
    double sum = 0;
    final int len = (estimator.length / 4) * 4;

    for (int i = 0; i < len; i += 4) {
      double diff0 = thetas[i] - estimator[i];
      double diff1 = thetas[i + 1] - estimator[i + 1];
      double diff2 = thetas[i + 2] - estimator[i + 2];
      double diff3 = thetas[i + 3] - estimator[i + 3];
      diff0 *= diff0;
      diff1 *= diff1;
      diff2 *= diff2;
      diff3 *= diff3;
      diff0 += diff2;
      diff1 += diff3;
      sum += diff0 + diff1;
    }
    for (int i = len; i < estimator.length; ++i) {
      sum += sqr(thetas[i] - estimator[i]);
    }
    return sum;
  }

  public final double naiveQuality() {
    double sum = 0;
    final int len = (thetas.length / 4) * 4;
    for (int i = 0; i < len; i += 4) {
      double p0 = sums[i] * 1.0 / n;
      double p1 = sums[i + 1] * 1.0 / n;
      double p2 = sums[i + 2] * 1.0 / n;
      double p3 = sums[i + 3] * 1.0 / n;
      double diff0 = thetas[i] - p0;
      double diff1 = thetas[i + 1] - p1;
      double diff2 = thetas[i + 2] - p2;
      double diff3 = thetas[i + 3] - p3;
      diff0 *= diff0;
      diff1 *= diff1;
      diff2 *= diff2;
      diff3 *= diff3;
      diff0 += diff2;
      diff1 += diff3;
      sum += diff0 + diff1;
    }
    for (int i = len; i < thetas.length; ++i) {
      sum += sqr(thetas[i] - sums[i] * 1.0 / n);
    }
    return sum;
  }


  public double[] naive() {
    double est[] = new double[sums.length];
    final int len = (thetas.length / 4) * 4;
    for (int i = 0; i < len; i += 4) {
      est[i] = sums[i] * 1.0 / n;
      est[i+1] = sums[i + 1] * 1.0 / n;
      est[i+2] = sums[i + 2] * 1.0 / n;
      est[i+3] = sums[i + 3] * 1.0 / n;
    }
    for (int i=len;i < thetas.length;++i)
      est[i] = sums[i] * 1.0 / n;
    return est;
  }
}
