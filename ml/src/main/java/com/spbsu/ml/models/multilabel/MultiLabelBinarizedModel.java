package com.spbsu.ml.models.multilabel;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: qdeee
 * Date: 01.04.15
 */
public class MultiLabelBinarizedModel extends MultiLabelModel.Stub {
  private final FuncJoin model;

  public MultiLabelBinarizedModel(final FuncJoin model) {
    this.model = model;
  }

  public MultiLabelBinarizedModel(final Func[] funcs) {
    this.model = new FuncJoin(funcs);
  }

  public FuncJoin getInternModel() {
    return model;
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

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    final MultiLabelBinarizedModel that = (MultiLabelBinarizedModel) o;

    if (!model.equals(that.model)) return false;

    return true;
  }

  @Override
  public int hashCode() {
    return model.hashCode();
  }
}
