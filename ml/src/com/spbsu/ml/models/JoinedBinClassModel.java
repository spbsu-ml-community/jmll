package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;

/**
 * User: qdeee
 * Date: 14.04.14
 * Description: this class is similar to MultiClassModel, however for JoinedBinClassModel it's assuming that you have
 * pack of INDEPENDENT binary classifier and their count equals to classes count.
 */
public class JoinedBinClassModel extends MCModel {
  public JoinedBinClassModel(final Func[] dirs) {
    super(dirs);
  }

  @Override
  public int classesCount() {
    return dirs.length;
  }

  @Override
  public double prob(int classNo, Vec x) {
    Vec apply = trans(x);
    return MathTools.sigmoid(apply.get(classNo), 0.65);
  }

  @Override
  public Vec probs(final Vec x) {
    Vec apply = trans(x);
    Vec probs = new ArrayVec(apply.dim());
    for (int i = 0; i < apply.dim(); i++) {
      probs.set(i, MathTools.sigmoid(apply.get(i), 0.65));
    }
    return probs;
  }

  @Override
  public int bestClass(Vec x) {
    double[] trans = trans(x).toArray();
    return ArrayTools.max(trans);
  }
}
