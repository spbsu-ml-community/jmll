package com.spbsu.ml.loss.multilabel;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.Func;
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
    double total = 0;
    for (int j = 0; j < targets.columns(); j++) {
      final Vec predictedLabels = predictMx.col(j);
      final Vec targetLabels = targets.col(j);
      final double v = VecTools.multiply(predictedLabels, targetLabels) / (VecTools.sum(predictedLabels) + VecTools.sum(targetLabels));
      if (!Double.isNaN(v)) {
        total += v;
      }
    }
    return 2 * total / targets.columns();
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
