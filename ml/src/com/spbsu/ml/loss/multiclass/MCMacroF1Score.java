package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.DataTools;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMacroF1Score extends Func.Stub {
  private final MCMacroPrecision precision;
  private final MCMacroRecall recall;

  public MCMacroF1Score(final Vec target) {
    precision = new MCMacroPrecision(target);
    recall = new MCMacroRecall(target);
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
}
