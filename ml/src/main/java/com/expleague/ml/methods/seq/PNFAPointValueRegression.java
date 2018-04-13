package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;

public class PNFAPointValueRegression extends FuncC1.Stub {
  private final Vec y;
  private final PNFAItemVecRegression regression;

  public PNFAPointValueRegression(Vec y, PNFAItemVecRegression regression) {
    this.regression = regression;
    this.y = y;
  }

  @Override
  public Vec gradientTo(Vec params, Vec to) {
    final Vec distribution = regression.distribution(params);
    final Vec r = VecTools.subtract(
        MxTools.multiply(
            regression.getValues(params), distribution
        ), y
    );
    final Mx grad = regression.getValues(to);
    final int stateCount = regression.stateCount();
    final int stateDim = regression.stateDim();
    for (int s = 0; s < stateCount; s++) {
      for (int i = 0; i < stateDim; i++) {
        final int idx = s * stateDim + i;
        grad.adjust(i, s, 2 * r.get(i) * distribution.get(s));
      }
    }

    return to;
  }

  @Override
  public double value(Vec params) {
    return regression.value(params);
  }

  @Override
  public int dim() {
    return (2 * regression.stateCount()) * regression.alphabetSize() + regression.stateCount() * regression.stateDim();
  }
}
