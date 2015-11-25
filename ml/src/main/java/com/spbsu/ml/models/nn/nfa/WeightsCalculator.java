package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ThreadLocalArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;

/**
* User: solar
* Date: 29.06.15
* Time: 17:12
*/
public class WeightsCalculator implements Computable<Vec,Mx> {
  protected final int statesCount;
  protected final int finalStates;
  private final int wStart;
  protected final int wLen;
  protected boolean[] dropOut;

  public WeightsCalculator(int statesCount, int finalStates, int wStart, int wLen) {
    this.statesCount = statesCount;
    this.finalStates = finalStates;
    this.wStart = wStart;
    this.wLen = wLen;
  }

  final ThreadLocalArrayVec w = new ThreadLocalArrayVec();
  public Mx computeInner(Vec betta) {
    final VecBasedMx b = new VecBasedMx(statesCount - 1, betta.sub(wStart, wLen));
    final VecBasedMx w = new VecBasedMx(statesCount, this.w.get(statesCount * statesCount));
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

  protected Vec cacheArg;
  protected Mx cacheVal;
  @Override
  public Mx compute(Vec betta) {
    if (!betta.isImmutable())
      return computeInner(betta);
    if (betta == cacheArg)
      return cacheVal;
    cacheArg = betta;
    return cacheVal = computeInner(betta);
  }

  public void setDropOut(boolean[] dropOut) {
    cacheArg = null;
    this.dropOut = dropOut;
  }
}
