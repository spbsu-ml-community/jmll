package com.spbsu.ml.loss.multilabel;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.data.set.DataSet;

/**
 * User: qdeee
 * Date: 20.03.15
 */
public class MultiLabelExactMatch extends Func.Stub implements ClassicMultiLabelLoss {
  private final Mx targets;

  public MultiLabelExactMatch(final Mx targets) {
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
      if (VecTools.distance(predictMx.row(i), targets.row(i)) < MathTools.EPSILON) {
        count++;
      }
    }
    return (double) count / targets.rows();
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
