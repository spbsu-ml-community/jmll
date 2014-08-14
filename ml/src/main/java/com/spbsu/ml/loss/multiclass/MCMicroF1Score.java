package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.Func;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMicroF1Score extends Func.Stub implements TargetFunc {
  private final MCMicroPrecision precision;
  private final MCMicroRecall recall;

  public MCMicroF1Score(final IntSeq target, DataSet<?> owner) {
    precision = new MCMicroPrecision(target, owner);
    recall = new MCMicroRecall(target, owner);
  }

  public MCMicroF1Score(Vec target, DataSet<?> owner) {
    precision = new MCMicroPrecision(target, owner);
    recall = new MCMicroRecall(target, owner);
  }

  @Override
  public double value(final Vec x) {
    double p = precision.value(x);
    double r = recall.value(x);
    return (p + r) > 0 ? 2 * p * r / (p + r) : 0.;
  }

  @Override
  public int dim() {
    return precision.dim();
  }

  @Override
  public DataSet<?> owner() {
    return precision.owner();
  }
}
