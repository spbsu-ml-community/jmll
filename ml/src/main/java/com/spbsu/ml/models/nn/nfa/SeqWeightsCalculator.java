package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;

/**
 * Created by afonin.s on 11.11.2015.
 */
public class SeqWeightsCalculator extends WeightsCalculator {
  private int[] wStarts;

  public SeqWeightsCalculator(int statesCount, int finalStates, int wLen, int... wStarts) {
    super(statesCount, finalStates, wStarts[0], wLen);
    this.wStarts = wStarts;
  }

  public Mx computeSeqInner(Vec betta) {
    Mx result = computeInner(betta, wStarts[0], wLen);
    for (int i = 1; i < wStarts.length; i++) {
      Mx mx = computeInner(betta, wStarts[i], wLen);
      result = MxTools.multiply(mx, result);
    }
    return result;
  }


  public Mx computeInner(Vec betta, int wStart, int wLen) {
    final VecBasedMx b = new VecBasedMx(statesCount - 1, betta.sub(wStart, wLen));
    final VecBasedMx w = new VecBasedMx(statesCount, new SparseVec(statesCount * statesCount));
    for (int i = 0; i < statesCount - finalStates; i++) {
      if (dropOut[i])
        continue;
      double sum = 1;
      for (int j = 0; j < statesCount - 1; j++) {
        if (dropOut[j])
          continue;
        sum += Math.exp(b.get(i, j));
      }
      for (int j = 0; j < statesCount; j++) {
        if (dropOut[j])
          continue;
        final double selectedExp = j < statesCount - 1 ? Math.exp(b.get(i, j)) : 1;
        w.set(j, i, selectedExp / sum);
      }
    }
    return w;
  }

  @Override
  public Mx compute(Vec betta) {
    if (!betta.isImmutable())
      return computeSeqInner(betta);
    if (betta == cacheArg)
      return cacheVal;
    cacheArg = betta;
    return cacheVal = computeSeqInner(betta);
  }


  public int[] getwStarts() {
    return wStarts;
  }
}
