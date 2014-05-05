package com.spbsu.ml.models;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
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

  public abstract int classesCount();
  public abstract double prob(int classNo, Vec x);
  public abstract Vec probs(Vec x);
  public abstract int bestClass(Vec x);

  public Vec probAll(int classNo, Mx data) {
    Vec result = new ArrayVec(data.rows());
    for (int i = 0; i < data.rows(); i++) {
      result.set(i, prob(classNo, data.row(i)));
    }
    return result;
  }

  public Mx probsAll(Mx data) {
    Mx result = new VecBasedMx(data.rows(), classesCount());
    for (int i = 0; i < result.rows(); i++) {
      Vec probs = probs(data.row(i));
      VecTools.assign(result.row(i), probs);
    }
    return result;
  }

  public Vec bestClassAll(Mx data) {
    Vec result = new ArrayVec(data.rows());
    for (int i = 0; i < data.rows(); i++) {
      result.set(i, bestClass(data.row(i)));
    }
    return result;
  }
}
