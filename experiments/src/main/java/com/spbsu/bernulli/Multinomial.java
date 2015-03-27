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

  public Multinomial(FastRandom rand, double[] q) {
    this.rand = rand;
    cum = new double[q.length];
    cum[0] = 0;
    for (int i = 1; i < q.length; ++i) {
      cum[i] = cum[i - 1] + q[i - 1];
    }
  }

  public int next() {
    final double coin = rand.nextDouble();
    return Math.abs(Arrays.binarySearch(cum, coin)) - 2;
  }
}
