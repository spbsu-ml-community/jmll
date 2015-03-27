package com.spbsu.bernulli;

import com.spbsu.commons.random.FastRandom;

import static java.util.Arrays.sort;

public abstract class Mixture {
  public final double[] q;
  protected final FastRandom random;

  protected Mixture(final double[] q, final FastRandom rand) {
    this.q = q;
    this.random = rand;
  }

  protected Mixture(int k,final FastRandom rand) {
    q = new double[k];
    this.random = rand;
    double total = 0;
    for (int i = 0; i < q.length; ++i) {
      q[i] = random.nextDouble();
      total += q[i];
    }
    for (int i = 0; i < q.length; ++i) {
      q[i] /= total;
    }
    sort(q);
  }

  public abstract MixtureObservations sample(int n);
}
