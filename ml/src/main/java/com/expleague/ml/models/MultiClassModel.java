package com.expleague.ml.models;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.models.multiclass.MCModel;
import com.expleague.commons.util.ArrayTools;

import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 10.01.14
 * Time: 10:59
 */
public class MultiClassModel extends MCModel.Stub {
  protected final Trans model;

  public MultiClassModel(final Func[] dirs) {
    this.model = new FuncJoin(dirs);
  }

  public MultiClassModel(final Trans model) {
    this.model = model;
  }

  public Trans getInternModel() {
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
  public Vec bestClassAll(final Mx x, final boolean parallel) {
    if (!parallel) {
      return bestClassAll(x);
    }

    Mx result = new VecBasedMx(1, new ArrayVec(x.rows()));
    IntStream.range(0, x.rows()).parallel().forEach(i -> result.set(i, this.value(x.row(i))));
    return result;
  }

  @Override
  public int dim() {
    return model.xdim();
  }
}
