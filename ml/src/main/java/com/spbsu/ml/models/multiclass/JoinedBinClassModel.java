package com.spbsu.ml.models.multiclass;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.func.FuncJoin;

/**
 * User: qdeee
 * Date: 14.04.14
 * Description: this class is similar to MultiClassModel, however for JoinedBinClassModel it's assuming that you have
 * pack of INDEPENDENT binary classifier and their count equals to classes count.
 */
public class JoinedBinClassModel extends MCModel.Stub {
  protected final FuncJoin internalModel;

  public JoinedBinClassModel(final Func[] dirs) {
    internalModel = new FuncJoin(dirs);
  }

  public JoinedBinClassModel(final FuncJoin internalModel) {
    this.internalModel = internalModel;
  }

  public FuncJoin getInternModel() {
    return internalModel;
  }

  @Override
  public int countClasses() {
    return internalModel.ydim();
  }

  @Override
  public Vec probs(final Vec x) {
    final Vec apply = internalModel.trans(x);
    final Vec probs = new ArrayVec(apply.dim());
    for (int i = 0; i < apply.dim(); i++) {
      probs.set(i, MathTools.sigmoid(apply.get(i)));
    }
    return probs;
  }

  @Override
  public int bestClass(final Vec x) {
    final double[] trans = internalModel.trans(x).toArray();
    return ArrayTools.max(trans);
  }

  @Override
  public int dim() {
    return internalModel.dirs[0].xdim();
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    final JoinedBinClassModel that = (JoinedBinClassModel) o;

    if (!internalModel.equals(that.internalModel)) return false;

    return true;
  }

  @Override
  public int hashCode() {
    return internalModel.hashCode();
  }
}
