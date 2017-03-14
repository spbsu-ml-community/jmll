package com.spbsu.ml.loss.multilabel;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 20.03.15
 *
 * expected {0,1} data
 */
public class MultiLabelHammingLoss extends Func.Stub implements ClassicMultiLabelLoss {
  private final Mx targets;

  public MultiLabelHammingLoss(final Mx targets) {
    this.targets = targets;
  }

  @Override
  public Mx getTargets() {
    return targets;
  }

  @Override
  public double value(final Vec x) {
    final Mx predictMx = (Mx) x;
    int count = 0;
    for (int i = 0; i < predictMx.rows(); i++) {
      for (int j = 0; j < predictMx.columns(); j++) {
        if (Math.abs(targets.get(i, j) - predictMx.get(i, j)) > MathTools.EPSILON) {
          count++;
        }
      }
    }
    return (double) count / (targets.rows() * targets.columns());
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
