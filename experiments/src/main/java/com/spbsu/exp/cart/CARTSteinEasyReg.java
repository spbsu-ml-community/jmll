package com.spbsu.exp.cart;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.L2Reg;

public class CARTSteinEasyReg extends L2Reg {
  public CARTSteinEasyReg(Vec target, DataSet<?> owner) {
    super(target, owner);
  }

  @Override
  public double value(final MSEStats stats) {
    return stats.weight >= 1 ? stats.sum / (stats.weight + 1.) : 0;
  }

  @Override
  public double score(final MSEStats stats) {
    final double weight = stats.weight;
    final double sum = stats.sum;
    return weight > 2 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;
  }

  @Override
  public double bestIncrement(final L2.MSEStats stats) {
//    if (stats.weight <= 2 || stats.sum2 < 1e-6)
//      return super.bestIncrement(stats);
    return stats.sum / (stats.weight + 1.);
  }
}
