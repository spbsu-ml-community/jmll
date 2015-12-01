package com.spbsu.ml.loss.multiclass;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 09.04.14
 */
public class MCMacroF1Score extends Func.Stub implements ClassicMulticlassLoss {
  private final MCMacroPrecision precision;
  private final MCMacroRecall recall;

  public MCMacroF1Score(final IntSeq target, final DataSet<?> owner) {
    precision = new MCMacroPrecision(target, owner);
    recall = new MCMacroRecall(target, owner);
  }

  public MCMacroF1Score(final Vec target, final DataSet<?> owner) {
    precision = new MCMacroPrecision(target, owner);
    recall = new MCMacroRecall(target, owner);
  }

  @Override
  public double value(final Vec x) {
    final double p = precision.value(x);
    final double r = recall.value(x);
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

  @Override
  public IntSeq labels() {
    return precision.labels();
  }
}
