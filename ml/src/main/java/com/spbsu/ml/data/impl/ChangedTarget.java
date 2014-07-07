package com.spbsu.ml.data.impl;

import com.spbsu.commons.math.vectors.Vec;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:09
 */
public class ChangedTarget extends LightDataSetImpl {
  final LightDataSetImpl parent;

  public ChangedTarget(LightDataSetImpl parent, Vec target) {
    super(parent.data(), target);
    this.parent = parent;
  }

  public int[] order(int fIndex) {
    return parent.order(fIndex);
  }
}
