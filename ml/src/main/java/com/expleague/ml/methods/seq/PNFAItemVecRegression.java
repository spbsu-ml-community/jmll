package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.methods.seq.param.BettaParametrization;
import com.expleague.ml.methods.seq.param.WeightParametrization;

public class PNFAItemVecRegression extends FuncC1.Stub {
  private final IntSeq seq;
  private final Vec y;
  private final int stateCount;
  private final int alphabetSize;
  private final int stateDim;
  private final BettaParametrization bettaParametrization;
  private final WeightParametrization weightParametrization;

  public PNFAItemVecRegression(final IntSeq seq,
                               final Vec y,
                               int stateCount,
                               int alphabetSize,
                               int stateDim,
                               BettaParametrization bettaParametrization,
                               WeightParametrization weightParametrization) {
    this.seq = seq;
    this.y = y;

    this.stateCount = stateCount;
    this.alphabetSize = alphabetSize;
    this.stateDim = stateDim;

    this.bettaParametrization = bettaParametrization;
    this.weightParametrization = weightParametrization;

  }

  @Override
  public int dim() {
    return bettaParametrization.paramCount(stateCount) * alphabetSize + stateCount * stateDim;
  }

  @Override
  public double value(Vec betta) {
    return VecTools.sum2(VecTools.subtract(vecValue(betta), y));
  }

  @Override
  public Vec gradientTo(Vec x, Vec grad) {
    VecTools.fill(grad, 0);
    final Vec state = new ArrayVec(stateCount * (seq.length() + 1));
    VecTools.fill(state.sub(0, stateCount), 1.0 / stateCount);
    //System.out.println("CPU Distribution: " + Arrays.toString(distribution.toArray()));
    for (int i = 0; i < seq.length(); i++) {
      Mx weightMx = weightParametrization.getMx(x, seq.intAt(i), stateCount);
      MxTools.multiplyTo(weightMx, state.sub(i * stateCount, stateCount), state.sub((i + 1) * stateCount, stateCount));
    }
    final Mx V = getValues(x);
    final Vec[] dS = new Vec[]{new ArrayVec(stateCount), new ArrayVec((stateCount))};
    final Vec lastLayerGrad = dS[seq.length() % 2];
    final Vec lastLayerState = state.sub(seq.length() * stateCount, stateCount);
    Vec r = MxTools.multiply(V, lastLayerState);
    VecTools.incscale(r, y, -1);

    final Mx vGrad = getValues(grad);
    for (int s = 0; s < stateCount; s++) {
      for (int i = 0; i < stateDim; i++) {
        final int idx = s * stateDim + i;
        vGrad.adjust(i, s, 2 * r.get(i) * lastLayerState.get(s));
      }
    }

    for (int s = 0; s < stateCount; s++) {
      double sum = 0;
      for (int d = 0; d < stateDim; d++) {
        sum += V.get(d, s) * r.get(d);
      }
      lastLayerGrad.set(s, 2 * sum);
    }

    final Mx dW = new VecBasedMx(stateCount, stateCount);
    final int paramCount = bettaParametrization.paramCount(stateCount);

    for (int i = seq.length() - 1; i >= 0; i--) {
      final int c = seq.intAt(i);
      Vec out = dS[(i + 1) % 2];
      Vec in = dS[i % 2];

      weightParametrization.gradientTo(x, state.sub(i * stateCount, stateCount), out, in, dW, c, stateCount);
      bettaParametrization.gradientTo(x, dW, grad.sub(paramCount * c, paramCount), c, stateCount);
    }
//    final double betta = 0.1 * 2 / stateCount / (stateCount - 1) / stateDim;
//    for (int i = 0; i < stateCount; i++) {
//      for (int j = i + 1; j < stateCount; j++) {
//        for (int c = 0; c < stateDim; c++) {
//          vGrad.adjust(c, i,-2 * betta * (V.get(i, c) - V.get(j, c)));
//          vGrad.adjust(c, j, -2 * betta * (V.get(j, c) - V.get(i, c)));
//        }
//      }
//    }
    return grad;
  }

  public Mx getValues(final Vec params) {
    return new VecBasedMx(
        stateCount,
        params.sub(params.dim() - stateCount * stateDim, stateCount * stateDim)
    );
  }

  public Vec distribution(Vec betta) {
    Vec[] distribution = new Vec[] {new ArrayVec(stateCount), new ArrayVec(stateCount)};
    VecTools.fill(distribution[0], 1.0 / stateCount);
    //System.out.println("CPU Distribution: " + Arrays.toString(distribution.toArray()));
    for (int i = 0; i < seq.length(); i++) {
      Mx weightMx = weightParametrization.getMx(betta, seq.intAt(i), stateCount);
      //System.out.println(String.format("-- (%s) CPU WeightMx: %s", i,
      //    Arrays.toString(weightMx.toArray())));
      MxTools.multiplyTo(weightMx, distribution[i % 2], distribution[(i + 1) % 2]);
    }

    return distribution[seq.length() % 2];
  }

  public Vec vecValue(Vec betta) {
    return MxTools.multiply(
        getValues(betta),
        distribution(betta)
    );
  }

  public int stateCount() {
    return stateCount;
  }

  public int stateDim() {
    return stateDim;
  }

  public int alphabetSize() {
    return alphabetSize;
  }


}


