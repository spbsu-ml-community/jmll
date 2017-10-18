package com.expleague.ml.loss.multilabel;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.Func;
import com.expleague.ml.data.set.DataSet;

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
    final double v = VecTools.sum(targets) + VecTools.sum(predictMx);
    if (v <= 0)
      return 0;
    return 2 * VecTools.multiply(targets, predictMx) / v;
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
