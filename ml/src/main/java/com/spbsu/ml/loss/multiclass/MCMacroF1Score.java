package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.TargetFunc;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMacroF1Score extends Func.Stub implements TargetFunc{
  private final MCMacroPrecision precision;
  private final MCMacroRecall recall;

  public MCMacroF1Score(final Vec target, DataSet<?> owner) {
    precision = new MCMacroPrecision(target, owner);
    recall = new MCMacroRecall(target, owner);
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
