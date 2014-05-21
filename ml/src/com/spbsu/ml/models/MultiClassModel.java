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
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: solar
 * Date: 10.01.14
 * Time: 10:59
 */
public class MultiClassModel extends MCModel {
  public MultiClassModel(Func[] dirs) {
    super(dirs);
  }

  @Override
  public int classesCount() {
    return dirs.length + 1;
  }

  @Override
  public double prob(int classNo, Vec x) {
    Vec trans = trans(x);
    double sumExps = 0.;
    for (int i = 0; i < trans.dim(); i++)
      sumExps += Math.exp(trans.get(i));
    return classNo < dirs.length? Math.exp(trans.get(classNo)) / (sumExps + 1.)
                                : 1. - sumExps / (sumExps + 1.);
  }

  @Override
  public Vec probs(final Vec x) {
    final Vec result = new ArrayVec(dirs.length + 1);
    final Vec trans = trans(x);
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
   * @param x features vector
   */
  @Override
  public int bestClass(final Vec x) {
    double[] trans = trans(x).toArray();
    int bestClass = ArrayTools.max(trans);
    return trans[bestClass] > 0 ? bestClass : dirs.length;
  }

  public static MultiClassModel joinBoostingResults(Ensemble ensemble) {
    if (ensemble.last() instanceof MCModel) {
      final Func[] joinedModels = new Func[ensemble.ydim()];
      final Func[][] transpose = new Func[ensemble.ydim()][ensemble.size()];
      for (int c = 0; c < transpose.length; c++) {
        for (int iter = 0; iter < transpose[c].length; iter++) {
          final MultiClassModel mcm = (MultiClassModel) ensemble.models[iter];
          transpose[c][iter] = mcm.dirs()[c];
        }
      }
      for (int i = 0; i < joinedModels.length; i++) {
        joinedModels[i] = new FuncEnsemble(transpose[i], ensemble.weights);
      }
      return new MultiClassModel(joinedModels);
    }
    else
      throw new ClassCastException("Ensemble object does not contain MultiClassModel objects");
  }
}
