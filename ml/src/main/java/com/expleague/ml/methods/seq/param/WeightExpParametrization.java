package com.expleague.ml.methods.seq.param;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;

public class WeightExpParametrization implements WeightParametrization{
  private final BettaParametrization bettaParametrization;

  public WeightExpParametrization(BettaParametrization bettaParametrization) {
    this.bettaParametrization = bettaParametrization;
  }

  @Override
  public Mx getMx(Vec params, int c, int stateCount) {
    Mx betta = bettaParametrization.getBettaMx(params, c, stateCount);
    return getMx(betta, c, stateCount);
  }

  private Mx getMx(Mx betta, int c, int stateCount) {
    Mx result = new VecBasedMx(stateCount, stateCount);

    for (int f = 0; f < stateCount; f++) {
      double sum = 0;
      for (int j = 0; j < stateCount; j++) {
        final double e = Math.exp(betta.get(f, j));
        sum += e;
        result.set(j, f, e);
      }

      for (int t = 0; t < stateCount; t++) {
        result.set(t, f, result.get(t, f) / sum);
      }
    }

    return result;
  }

  @Override
  public void gradientTo(Vec params, Vec out, Vec dOut, Vec dOutNew, Mx dW, int c, int stateCount) {
    VecTools.fill(dW, 0);

    final Mx betta = bettaParametrization.getBettaMx(params, c, stateCount);
    final Mx W = getMx(betta, c, stateCount);

    for (int f = 0; f < stateCount; f++) {
      final double prevS_f = out.get(f);

      double sum = 0;
      for (int i = 0; i < stateCount; i++) {
        sum += Math.exp(betta.get(f, i));
      }

      double prevdS_f = 0;
      for (int t = 0; t < stateCount; t++) {
        { // dT/dS_prev_f
          prevdS_f += W.get(t, f) * dOut.get(t);
        }

        { // dT/dM
          final double grad = prevS_f * dOut.get(t);
          final double betta_ft = betta.get(f, t);
          final double betta_ft_exp = Math.exp(betta_ft);
          final double multiply = 2 * grad  / sum / sum;

          for (int j = 0; j < stateCount; j++) {
            final double betta_fj = betta.get(f, j);
            if (j == t) {
              dW.adjust(f, j, grad * W.get(j, f) * (1 - W.get(j, f)));
            }
            else {
              dW.adjust(f, j, -grad * W.get(j, f)* W.get(t, f));
            }
          }
        }
      }
      dOutNew.set(f, prevdS_f);
    }
  }
}
