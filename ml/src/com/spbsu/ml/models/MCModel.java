package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Func;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: qdeee
 * Date: 14.04.14
 */
public abstract class MCModel extends FuncJoin{
  protected MCModel(final Func[] dirs) {
    super(dirs);
  }

  public static MultiClassModel joinBoostingResults(Ensemble ensemble) {
    if (ensemble.last() instanceof MCModel) {
      Func[] joinedModels;
      Func[][] transpose = new Func[ensemble.ydim()][ensemble.size()];
      for (int c = 0; c < transpose.length; c++) {
        for (int iter = 0; iter < transpose[c].length; iter++) {
          final MultiClassModel mcm = (MultiClassModel) ensemble.models[iter];
          transpose[c][iter] = mcm.dirs()[c];
        }
      }
      joinedModels = new Func[ensemble.ydim()];
      for (int i = 0; i < joinedModels.length; i++) {
        joinedModels[i] = new FuncEnsemble(transpose[i], ensemble.weights);
      }
      return new MultiClassModel(joinedModels);
    }
    else
      throw new ClassCastException("Ensemble object does not contain MultiClassModel objects");
  }

  public abstract double p(int classNo, Vec x);

  public abstract int bestClass(Vec x);

  public Vec bestClassAll(Mx data) {
    Vec result = new ArrayVec(data.rows());
    for (int i = 0; i < data.rows(); i++) {
      result.set(i, bestClass(data.row(i)));
    }
    return result;
  }
}
