package com.spbsu.ml.models.multilabel;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: qdeee
 * Date: 01.04.15
 */
public class LogitMultiLabelModel extends MultiLabelModel.Stub {
  private final FuncJoin model;

  public LogitMultiLabelModel(final FuncJoin model) {
    this.model = model;
  }

  @Override
  public Vec predictLabels(final Vec x) {
    final Vec trans = model.trans(x);
    return VecTools.toBinary(trans, 0);
  }

  @Override
  public int xdim() {
    return model.xdim();
  }

  @Override
  public int ydim() {
    return model.ydim();
  }
}
