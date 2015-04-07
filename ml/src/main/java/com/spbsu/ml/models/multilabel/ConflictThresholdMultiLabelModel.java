package com.spbsu.ml.models.multilabel;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.models.multiclass.MCModel;

/**
 * User: qdeee
 * Date: 22.03.15
 */
public class ConflictThresholdMultiLabelModel extends MultiLabelModel.Stub {
  private final MCModel intern;
  private final double threshold;
  private final boolean allZeroesClassEnabled;

  public ConflictThresholdMultiLabelModel(final MCModel intern, final double threshold, final boolean allZeroesClassEnabled) {
    this.intern = intern;
    this.threshold = threshold;
    this.allZeroesClassEnabled = allZeroesClassEnabled;
  }

  @Override
  public Vec predictLabels(final Vec x) {
    final Vec prediction = intern.probs(x);
    final int argMax = VecTools.argmax(prediction);
    VecTools.toBinary(prediction, threshold);
    if (allZeroesClassEnabled) {
      if (argMax == prediction.dim() - 1) {
        //all zeroes class
        return new ArrayVec(ydim());
      } else {
        return prediction.sub(0, ydim());
      }
    } else {
      return prediction;
    }
  }

  @Override
  public int xdim() {
    return intern.xdim();
  }

  @Override
  public int ydim() {
    return allZeroesClassEnabled ? intern.countClasses() - 1 : intern.countClasses();
  }
}
