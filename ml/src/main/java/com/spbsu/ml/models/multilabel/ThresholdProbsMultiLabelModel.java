package com.spbsu.ml.models.multilabel;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.models.multiclass.MCModel;

/**
 * User: qdeee
 * Date: 22.03.15
 */
public class ThresholdProbsMultiLabelModel extends MultiLabelModel.Stub {
  private final MCModel intern;
  private final double threshold;

  public ThresholdProbsMultiLabelModel(final MCModel intern, final double threshold) {
    this.intern = intern;
    this.threshold = threshold;
  }

  @Override
  public Vec predictLabels(final Vec x) {
    final Vec probs = intern.probs(x);
    return VecTools.toBinary(probs, threshold);
  }

  @Override
  public int xdim() {
    return intern.xdim();
  }

  @Override
  public int ydim() {
    return intern.countClasses();
  }
}
