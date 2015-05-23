package com.spbsu.bernulli;

import com.spbsu.commons.random.FastRandom;

import java.util.Arrays;

/**
 * User: Noxoomo
 * Date: 20.03.15
 * Time: 15:26
 */

public class Multinomial {
  private final double[] cum;
  private final FastRandom rand;
  private static final double eps = 1e-15;

  public Multinomial(FastRandom rand, double[] q) {
    this.rand = rand;
    double totalWeight = 0;
    cum = new double[q.length];
    cum[0] = 0;
    totalWeight = q[0];
    for (int i = 1; i < q.length; ++i) {
      cum[i] = cum[i - 1] + q[i - 1];
      totalWeight += q[i];
    }
    if (totalWeight < 1 - eps) {
      for (int i = 0; i < cum.length; ++i)
        cum[i] /= totalWeight;
    }
  }

  public int next() {
    final double coin = rand.nextDouble();
    return Math.abs(Arrays.binarySearch(cum, coin)) - 2;
  }

  public static int next(FastRandom rand, double[] weights) {
    double totalWeight;
    final double[] cum = new double[weights.length];
    cum[0] = 0.0;
    totalWeight = weights[0];
    for (int i = 1; i < weights.length; ++i) {
      cum[i] = cum[i - 1] + weights[i - 1];
      totalWeight += weights[i];
    }
    final double coin = totalWeight*rand.nextDouble();
    return upperBound(cum, coin) - 1;
  }

  static int upperBound(final double[] values, final double key) {
    int left = 0;
    int right = values.length;
    while (right - left > 24) {
      final int mid = (left + right) >>> 1;
      final double midVal = values[mid];
      if (midVal <= key)
        left = mid;
      else
        right = mid;
    }
    for (int i = left; i < right;++i) {
      if (values[i] > key)
        return i;
    }
    return right;
  }
}
