package com.expleague.ml.loss.multilabel;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.FuncC1;
import com.expleague.ml.TargetFunc;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.loss.LLLogit;

/**
 * User: qdeee
 * Date: 07.04.15
 */
public class MultiLabelOVRLogit extends FuncC1.Stub implements TargetFunc {
  private final Mx targets;
  private final LLLogit[] perColumnLogit;

  public MultiLabelOVRLogit(final Mx targets) {
    this.targets = targets;
    this.perColumnLogit = new LLLogit[targets.columns()];
    for (int j = 0; j < perColumnLogit.length; j++) {
      perColumnLogit[j] = new LLLogit(targets.col(j), null);
    }
  }

  @Override
  public Vec gradient(final Vec x) {
    final Mx mx = new VecBasedMx(perColumnLogit.length, x);

    final Vec[] colGrads = new Vec[perColumnLogit.length];
    for (int j = 0; j < colGrads.length; j++) {
      colGrads[j] = perColumnLogit[j].gradient(mx.col(j));
    }
    return new ColsVecArrayMx(colGrads);
  }

  @Override
  public DataSet<?> owner() {
    return null;
  }

  @Override
  public double value(final Vec x) {
    final Mx mx = x instanceof Mx
        ? (Mx) x
        : new VecBasedMx(targets.columns(), x);

    final Vec values = new ArrayVec(perColumnLogit.length);
    for (int j = 0; j < values.dim(); j++) {
      final double value = perColumnLogit[j].value(mx.col(j));
      System.out.println(value);
      values.set(j, value);
    }
    return MathTools.meanNaive(values);
  }

  @Override
  public int dim() {
    return targets.dim();
  }
}
