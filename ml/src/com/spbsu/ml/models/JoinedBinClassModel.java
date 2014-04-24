package com.spbsu.ml.models;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.Func;
import com.spbsu.ml.models.MCModel;

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
  public double p(int classNo, Vec x) {
    Vec apply = trans(x);
    double sumSigmoids = 0.;
    for (int i = 0; i < apply.dim(); i++)
      sumSigmoids += MathTools.sigmoid(apply.get(i));
    return MathTools.sigmoid(apply.get(classNo)) / sumSigmoids;
  }

  @Override
  public int bestClass(Vec x) {
    double[] trans = trans(x).toArray();
    return ArrayTools.max(trans);
  }
}
