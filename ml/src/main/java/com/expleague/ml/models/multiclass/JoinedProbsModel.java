package com.expleague.ml.models.multiclass;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.func.FuncJoin;

/**
 * User: qdeee
 * Date: 07.04.15
 */
public class JoinedProbsModel extends FuncJoin {
  public JoinedProbsModel(final Func[] dirs) {
    super(dirs);
  }

  @Override
  public Vec trans(final Vec x) {
    final Vec result = super.trans(x);
    for (int i = 0; i < result.dim(); i++) {
      result.set(i, MathTools.sigmoid(result.get(i)));
    }
    return result;
  }
}
