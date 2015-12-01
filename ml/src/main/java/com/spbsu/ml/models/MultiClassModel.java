package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.models.multiclass.MCModel;

/**
 * User: solar
 * Date: 10.01.14
 * Time: 10:59
 */
public class MultiClassModel extends MCModel.Stub {
  protected final FuncJoin model;

  public MultiClassModel(final Func[] dirs) {
    this.model = new FuncJoin(dirs);
  }

  public MultiClassModel(final FuncJoin model) {
    this.model = model;
  }

  public FuncJoin getInternModel() {
    return model;
  }

  @Override
  public int countClasses() {
    return model.ydim() + 1;
  }

  @Override
  public Vec probs(final Vec x) {
    final Vec result = new ArrayVec(model.ydim() + 1);
    final Vec trans = model.trans(x);
    double sumExps = 0.;
    for (int i = 0; i < trans.dim(); i++) {
      final double exp = Math.exp(trans.get(i));
      result.set(i, exp);
      sumExps += exp;
    }
    for (int i = 0; i < trans.dim(); i++) {
      result.set(i, result.get(i) / (sumExps + 1));
    }
    result.set(trans.dim(), 1 - sumExps / (sumExps + 1));
    return result;
  }

  /**
   * Def: 'sum_i^{N-1}{e^{s_i}}' as 'S'.
   * If we need to compare 'e^{s_k}/(S + 1}' and '1 - S/(S + 1)', it's enough to compare 's_k' and '0'.
   *
   * @param x lines vector
   */
  @Override
  public int bestClass(final Vec x) {
    final double[] trans = model.trans(x).toArray();
    final int bestClass = ArrayTools.max(trans);
    return trans[bestClass] > 0 ? bestClass : model.ydim();
  }

  @Override
  public int dim() {
    return model.xdim();
  }
}
