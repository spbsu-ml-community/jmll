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
public class MultiLabelMicroFScore extends Func.Stub implements ClassicMultiLabelLoss {
  private final Mx targets;

  public MultiLabelMicroFScore(final Mx targets) {
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
    for (int i = 0; i < predictMx.rows(); i++) {
      final Vec predictedLabels = predictMx.row(i);
      final Vec targetLabels = targets.row(i);
      final double v = VecTools.multiply(predictedLabels, targetLabels) / (VecTools.sum(predictedLabels) + VecTools.sum(targetLabels));
      if (!Double.isNaN(v)) {
        total += v;
      }
    }
    return 2 * total / targets.rows();
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
