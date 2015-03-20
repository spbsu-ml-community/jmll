package com.spbsu.ml.loss.multilabel;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 20.03.15
 *
 * expected {0,1} data
 */
public class MultiLabelMacroFScore extends Func.Stub implements ClassicMultiLabelLoss {
  private final Mx targets;

  public MultiLabelMacroFScore(final Mx targets) {
    this.targets = targets;
  }

  @Override
  public Mx getTargets() {
    return targets;
  }

  @Override
  public double value(final Vec x) {
    final Mx predictMx = (Mx) x;
    double value = 0;
    for (int j = 0; j < targets.columns(); j++) {
      final Vec predictedLabels = predictMx.col(j);
      final Vec targetLabels = targets.col(j);
      value += VecTools.multiply(predictedLabels, targetLabels) / (VecTools.sum(predictedLabels) + VecTools.sum(targetLabels));
    }
    return 2 * value / targets.rows();
  }

  @Override
  public int dim() {
    return targets.dim();
  }

  @Override
  public DataSet<?> owner() {
    return null;
  }
}
