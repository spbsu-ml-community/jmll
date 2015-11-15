package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;

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


  @Override
  public Mx compute(Vec betta) {
    if (!betta.isImmutable())
      return computeSeqInner(betta);
    if (betta == cacheArg)
      return cacheVal;
    cacheArg = betta;
    return cacheVal = computeSeqInner(betta);
  }

}
