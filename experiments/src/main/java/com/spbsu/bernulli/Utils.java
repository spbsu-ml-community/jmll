package com.spbsu.bernulli;

import static com.spbsu.commons.math.MathTools.sqr;

/**
 * Created by noxoomo on 11/04/15.
 */
public class Utils {

  public static double dist(double[] first, double[] second) {
    final int len = (first.length / 4) * 4;
    double sum = 0;
    for (int i = 0; i < len; i += 4) {
      double diff0 = first[i] - second[i];
      double diff1 = first[i + 1] - second[i + 1];
      double diff2 = first[i + 2] - second[i + 2];
      double diff3 = first[i + 3] - second[i + 3];
      diff0 *= diff0;
      diff1 *= diff1;
      diff2 *= diff2;
      diff3 *= diff3;
      diff0 += diff2;
      diff1 += diff3;
      sum += diff0 + diff1;
    }
    for (int i = len; i < first.length; ++i)
      sum += sqr(first[i] - second[i]);
    return sum;
  }
}
