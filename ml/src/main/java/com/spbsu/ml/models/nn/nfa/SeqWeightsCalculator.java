package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;

/**
 * Created by afonin.s on 11.11.2015.
 */
public class SeqWeightsCalculator extends WeightsCalculator {
  private int[] wStarts;
  private final ThreadLocalArrayVec[] w;

  public SeqWeightsCalculator(int statesCount, int finalStates, int wLen, int... wStarts) {
    super(statesCount, finalStates, wStarts[0], wLen);
    this.wStarts = wStarts;
    this.w = new ThreadLocalArrayVec[wStarts.length];
    for (int i = 0; i < wStarts.length; i++) {
      w[i] = new ThreadLocalArrayVec();
    }
  }

  private Mx computeSeqInner(Vec betta) {
    Mx result = computeInner(betta, wStarts[0], wLen, 0);
    for (int i = 1; i < wStarts.length; i++) {
      Mx mx = computeInner(betta, wStarts[i], wLen, i);
      result = MxTools.multiply(mx, result);
    }
    return result;
  }


  private Mx computeInner(Vec betta, int wStart, int wLen, int index) {
    final VecBasedMx b = new VecBasedMx(statesCount - 1, betta.sub(wStart, wLen));
    final VecBasedMx w = new VecBasedMx(statesCount, this.w[index].get(statesCount * statesCount));
    makeMatrix(b, w);
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

  public Vec gradientTo(Vec betta, Vec to, Vec subParents, int index) {
    final Mx weights = this.compute(betta);
    final int bettaDim = statesCount - 1;
    final int indexLocal = index;
    final VecBasedMx grad = new VecBasedMx(bettaDim, to.sub(wStarts[0], wLen));
    for (int i = 0; i < grad.rows(); i++) {
      final double selectedProbab = weights.get(indexLocal, i);
      for (int j = 0; j < bettaDim; j++) {
        double currentProbab = weights.get(j, i);
        if (j == indexLocal)
          grad.set(i, j, subParents.get(i) * selectedProbab * (1 - selectedProbab));
        else
          grad.set(i, j, -subParents.get(i) * selectedProbab * currentProbab);
      }
    }
    return to;
  }
}
