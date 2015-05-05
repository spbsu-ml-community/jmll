package com.spbsu.ml.models.multilabel;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.models.multiclass.MCModel;

/**
 * User: qdeee
 * Date: 23.03.15
 */
public class MultiLabelSubsetsModel extends MultiLabelModel.Stub {
  private final MCModel intern;
  private final Vec[] class2labels;

  public MultiLabelSubsetsModel(final MCModel intern, final Vec[] class2labels) {
    this.intern = intern;
    this.class2labels = class2labels;
  }

  @Override
  public Vec predictLabels(final Vec x) {
    final int bestClass = intern.bestClass(x);
    return class2labels[bestClass];
  }

  @Override
  public int xdim() {
    return intern.xdim();
  }

  @Override
  public int ydim() {
    return class2labels[0].dim();
  }
}
