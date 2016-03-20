package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;

/**
 * Created by afonin.s on 11.11.2015.
 */
public class SeqWeightsCalculator extends WeightsCalculator {
  private int[] wStarts;
  private final ThreadLocalArrayVec[] w;
  private final ThreadLocalArrayVec[] gradW;
  private final Vec[] cacheArgs;
  private final Mx[] cacheVals;

  public SeqWeightsCalculator(int statesCount, int finalStates, int wLen, int... wStarts) {
    super(statesCount, finalStates, wStarts[0], wLen);
    this.wStarts = wStarts;
    this.w = new ThreadLocalArrayVec[wStarts.length];
    this.gradW = new ThreadLocalArrayVec[wStarts.length];
    for (int i = 0; i < wStarts.length; i++) {
      w[i] = new ThreadLocalArrayVec();
      gradW[i] = new ThreadLocalArrayVec();
    }
    this.cacheArgs = new Vec[wStarts.length];
    this.cacheVals = new Mx[wStarts.length];
  }

  private Mx computeSeqInner(Vec betta) {
    Mx result = computeInner(betta, 0);
    for (int i = 1; i < wStarts.length; i++) {
      Mx mx = computeInner(betta, i);
      result = MxTools.multiply(mx, result);
    }
    return result;
  }


  private Mx computeInner(Vec betta, int index) {
    if (betta == cacheArgs[index]) {
      return cacheVals[index];
    }
    final VecBasedMx b = new VecBasedMx(statesCount - 1, betta.sub(wStarts[index], wLen));
    final VecBasedMx w = new VecBasedMx(statesCount, this.w[index].get(statesCount * statesCount));
    makeMatrix(b, w);
    cacheArgs[index] = betta;
    cacheVals[index] = w;
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
    int bettaDim = statesCount - 1;
    for (int r = 0; r < wStarts.length; r++) {
      for (int i0 = 0; i0 < statesCount - finalStates; i0++) {
        for (int j0 = 0; j0 < bettaDim; j0++) {
          to.set(wStarts[r] + i0 * bettaDim + j0, VecTools.multiply(computeGradInner(betta, r, i0, j0).row(index), subParents));
        }
      }
    }

    return to;
  }

  private Mx computeGradInner(Vec betta, int r0, int i0, int j0) {
    Mx result;
    if (r0 == 0)
      result = getGradMx(betta, 0, i0, j0);
    else
      result = this.computeInner(betta, 0);
    for (int r = 1; r < wStarts.length; r++) {
      Mx mx;
      if (r == r0)
        mx = getGradMx(betta, r, i0, j0);
      else
        mx = this.computeInner(betta, r);
      result = MxTools.multiply(mx, result);
    }
    return result;
  }

  private Mx getGradMx(Vec betta, int r, int i0, int j0) {
    final Mx weights = this.computeInner(betta, r);
    final VecBasedMx grad = new VecBasedMx(statesCount, gradW[r].get(statesCount * statesCount));

    final double selectedProbab = weights.get(j0, i0);
    for (int i = 0; i < statesCount; i++) {
      if (i == j0)
        grad.set(i, i0, selectedProbab * (1 - selectedProbab));
      else
        grad.set(i, i0, -selectedProbab * weights.get(i, i0));
    }
    return grad;
  }
}
