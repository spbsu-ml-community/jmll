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
    final int bettaDim = statesCount - 1;

    if (wStarts.length == 1) {
      final Mx grad = getGradMx(index, betta, bettaDim, 0);

      for (int i = 0; i < grad.rows(); i++) {
        for (int j = 0; j < grad.columns(); j++) {
          to.set(wStarts[0] + i * bettaDim + j, grad.get(i, j) * subParents.get(i));
        }
      }
    } else if (wStarts.length == 2) {
      for (int r = 0; r < 2; r++) {
        final Mx grad = computeGradInner(index, betta, subParents, bettaDim, r == 0);

        for (int i = 0; i < grad.rows(); i++) {
          for (int j = 0; j < grad.columns(); j++) {
            to.set(wStarts[r] + i * bettaDim + j, grad.get(i, j));
          }
        }
      }
    } else {
      throw new RuntimeException("wStarts.lenght > 2: " + wStarts.length);
    }
    return to;
  }

  private Mx computeGradInner(int gradIndex, Vec betta,  Vec subParents, int bettaDim, boolean first) {
    Mx result;
    if (first) {
      final Mx weights = this.computeInner(betta, 1);
      result = VecTools.scale(getGradMx(0, betta, bettaDim, 0), weights.get(gradIndex, 0));
      for (int j = 1; j < statesCount - finalStates; j++) {
        VecTools.append(result, VecTools.scale(getGradMx(j, betta, bettaDim, 0), weights.get(gradIndex, j)));
      }
      for (int i = 0; i < result.rows(); i++) {
        VecTools.scale(result.row(i), subParents.get(i));
      }
    } else {
      final Mx weights = this.computeInner(betta, 0);
      final Vec vec = MxTools.multiply(weights, subParents);
      result = getGradMx(gradIndex, betta, bettaDim, 1);
      for (int i = 0; i < result.rows(); i++) {
        VecTools.scale(result.row(i), vec.get(i));
      }
    }
    return result;
  }

  private Mx getGradMx(int gradIndex, Vec betta, int bettaDim, int index) {
    final Mx weights = this.computeInner(betta, index);
    final VecBasedMx grad = new VecBasedMx(bettaDim, gradW[index].get(wLen));
    for (int i = 0; i < grad.rows(); i++) {
      final double selectedProbab = weights.get(gradIndex, i);
      for (int j = 0; j < bettaDim; j++) {
        double currentProbab = weights.get(j, i);
        if (j == gradIndex)
          grad.set(i, j, selectedProbab * (1 - selectedProbab));
        else
          grad.set(i, j, -selectedProbab * currentProbab);
      }
    }
    return grad;
  }
}
