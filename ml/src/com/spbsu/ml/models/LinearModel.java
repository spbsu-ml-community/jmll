package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Model;

/**
 * User: solar
 * Date: 01.03.11
 * Time: 22:30
 */
public class LinearModel implements Model {
  protected final Vec betas;

  public LinearModel(Vec betas) {
    this.betas = betas;
  }

  public double value(Vec point) {
    return VecTools.multiply(betas, point);
  }

  public String toString() {
    String result = "";
    int[] order = ArrayTools.sequence(0, betas.dim());
    double[] abs = new double[betas.dim()];
    VecIterator it = betas.nonZeroes();
    while (it.advance())
      abs[it.index()] = Math.abs(it.value());
    ArrayTools.parallelSort(abs, order);
    for (int i = 0; i < order.length && abs[order[i]] > 0; i++) {
      result += "\t" + order[i] + ": " + betas.get(order[i]);
    }
    return result;
  }

  public final Vec weights() {
    return betas;
  }
}
