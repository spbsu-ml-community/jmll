package com.spbsu.ml.models;

import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: solar
 * Date: 10.01.14
 * Time: 10:59
 */
public class MultiClassModel extends FuncJoin {
  public MultiClassModel(Func[] dirs) {
    super(dirs);
  }

  public double p(int classNo, Vec x) {
    Vec trans = trans(x);
    double sumExps = 0.;
    for (int i = 0; i < trans.dim(); i++)
      sumExps += Math.exp(trans.get(i));
    return classNo < dirs.length? Math.exp(trans.get(classNo)) / (sumExps + 1.)
                                : 1. - sumExps / (sumExps + 1.);
  }

  /**
   * Def: 'sum_i^{N-1}{e^{s_i}}' as 'S'.
   * If we need to compare 'e^{s_k}/(S + 1}' and '1 - S/(S + 1)', it's enough to compare 's_k' and '0'.
   *
   * @param x features vector
   */
  public int bestClass(final Vec x) {
    double[] trans = trans(x).toArray();
    int bestClass = ArrayTools.max(trans);
    return trans[bestClass] > 0 ? bestClass : dirs.length;
  }

  public Vec bestClassAll(Mx data) {
    Vec result = new ArrayVec(data.rows());
    for (int i = 0; i < data.rows(); i++) {
      result.set(i, bestClass(data.row(i)));
    }
    return result;
  }
}
