package com.expleague.ml.methods.seq.param;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;

public class BettaTwoVecParametrization implements BettaParametrization {
  private final double addToDiag;

  public BettaTwoVecParametrization(double addToDiag) {
    this.addToDiag = addToDiag;
  }

  @Override
  public int paramCount(int stateCount) {
    return 2 * stateCount;
  }

  @Override
  public Mx getBettaMx(Vec params, int c, int stateCount) {
    int bettaSize = paramCount(stateCount);
    Mx betta = new VecBasedMx(stateCount, stateCount);

    for (int i = 0; i < stateCount; i++) {
      for (int j = 0; j < stateCount; j++) {
        double value = Math.min(1e9, Math.max(-1e9, params.get(c * bettaSize + i) * params.get(c * bettaSize + stateCount + j)));
        if (i == j) {
          value += addToDiag;
        }
        betta.set(i, j, value);
      }
    }

    return betta;
  }

  @Override
  public void gradientTo(Vec params, Mx dBetta, Vec dParam, int c, int stateCount) {
    int bettaSize = paramCount(stateCount);

    final Vec v = params.sub(c * bettaSize, stateCount);
    final Vec u = params.sub(c * bettaSize + stateCount, stateCount);

    final Vec dv = dParam.sub(0, stateCount);
    final Vec du = dParam.sub(stateCount, stateCount);

    for (int k = 0; k < dBetta.rows(); k++) {
      for (int s = 0; s < dBetta.columns(); s++) {
        double m = dBetta.get(k, s);
        du.adjust(s, v.get(k) * m);
        dv.adjust(k, u.get(s) * m);
      }
    }
  }
}
