package com.expleague.ml.methods.seq;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.RegularizerFunc;
import com.expleague.ml.methods.seq.param.BettaParametrization;

import java.util.stream.IntStream;

/**
 * Created by hrundelb on 29.01.19.
 */
public class MyRegularizer extends RegularizerFunc.Stub {
  private final PNFAItemVecRegression[] funcs;
  private final int stateCount;
  private final double alpha;
  private final double betta;
  private final Vec wCacheVec;
  private Vec prev;
  private final BettaParametrization bettaParametrization;
  private final int alphabetSize;
  private final int stateDim;


  public MyRegularizer(PNFAItemVecRegression[] funcs, Vec wCacheVec, int stateCount, double alpha, double betta,
                       BettaParametrization bettaParametrization, int alphabetSize, int stateDim) {
    this.funcs = funcs;
    this.wCacheVec = wCacheVec;
    this.stateCount = stateCount;
    this.alpha = alpha;
    this.betta = betta;
    this.bettaParametrization = bettaParametrization;
    this.alphabetSize = alphabetSize;
    this.stateDim = stateDim;
    prev = null;
    prev = new ArrayVec(funcs[0].dim());
  }

  @Override
  public double value(Vec x) {
    int paramCount = bettaParametrization.paramCount(stateCount);
    return alpha * VecTools.l1(x.sub(0, paramCount * alphabetSize)) + betta * VecTools.l2(x.sub(paramCount * alphabetSize, stateCount * stateDim));
  }

  @Override
  public int dim() {
    return bettaParametrization.paramCount(stateCount) * alphabetSize + stateCount * stateDim;
  }

  @Override
  public Vec project(Vec x) {
    Mx values = funcs[0].getValues(x);
    IntStream.range(0, x.length() - values.dim()).filter(idx -> prev.get(idx) != x.get(idx)).forEach(idx -> {
      final double val = x.get(idx);
      if (Math.abs(val) > alpha)
        x.adjust(idx, val > alpha ? -alpha : alpha);
      else
        x.set(idx, 0);
    });
    VecTools.assign(prev, x);
    VecTools.scale(values, values.dim() / (betta + values.dim()));
    if (wCacheVec != null) {
      VecTools.fill(wCacheVec, -1);
    }
    return x;
  }
}
