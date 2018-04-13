package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;

public class PNFAItemVecRegression extends FuncC1.Stub {
  private final IntSeq seq;
  private final Vec y;
  private final int stateCount;
  private final int alphabetSize;
  private final int stateDim;

  private final double diag;
  private volatile Mx[] wCache;

  public PNFAItemVecRegression(final IntSeq seq, final Vec y, final Mx[] wCache, int stateCount, int alphabetSize, int stateDim, double diag) {
    this.wCache = wCache;
    this.seq = seq;
    this.y = y;

    this.stateCount = stateCount;
    this.alphabetSize = alphabetSize;
    this.stateDim = stateDim;

    this.diag = diag;
  }

  @Override
  public int dim() {
    return 2 * stateCount * alphabetSize + stateCount * stateDim;
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
      Mx weightMx = weightMx(x, seq.intAt(i));
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

    final Mx dM = new VecBasedMx(stateCount, stateCount - 1);
    for (int i = seq.length() - 1; i >= 0; i--) {
      final int c = seq.intAt(i);
      Vec out = dS[(i + 1) % 2];
      Vec in = dS[i % 2];

      mySoftmaxGradMx(c, x, state.sub(i * stateCount, stateCount), out, in, dM);
      { // dM -> du & dv
        final int betaIdx = 2 * stateCount * c;
        final Vec v = x.sub(betaIdx, stateCount);
        final Vec u = x.sub(betaIdx + stateCount, stateCount);
        final Vec dv = grad.sub(betaIdx, stateCount);
        final Vec du = grad.sub(betaIdx + stateCount, stateCount);
        for (int k = 0; k < dM.rows(); k++) {
          for (int s = 0; s < dM.columns(); s++) {
            double m = dM.get(k, s);
            du.adjust(s, v.get(k) * m);
            dv.adjust(k, u.get(s) * m);
          }
        }
      }
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

  Mx getValues(final Vec params) {
    return new VecBasedMx(
        stateCount,
        params.sub(params.dim() - stateCount * stateDim, stateCount * stateDim)
    );
  }

  public Mx weightMx(final Vec betta, final int c) {
    if (wCache == null)
      return weightMx(betta, c, new VecBasedMx(stateCount, stateCount), stateCount, diag);
    final Mx wCached = wCache[c];
    if (wCached.get(wCached.dim() - 1) >= 0)
      return wCached;
    //noinspection SynchronizationOnLocalVariableOrMethodParameter
    synchronized (wCached) {
      if (wCached.get(wCached.dim() - 1) >= 0)
        return wCached;
      return weightMx(betta, c, wCached, stateCount, diag);
    }
  }

  public static Mx weightMx(Vec betta, int c, int stateCount, double diag) {
    return weightMx(betta, c, new VecBasedMx(stateCount, stateCount), stateCount, diag);
  }

  private static Mx weightMx(Vec betta, int c, Mx to, int stateCount, double diag) {
    for (int f = 0; f < stateCount; f++) {
      double sum = 1;
      for (int j = 0; j < stateCount - 1; j++) {
        final double bettaIJ = getBetta(betta, c, f, j, stateCount, diag);
        final double e = MathTools.sqr(bettaIJ);
        sum += e;
        to.set(j, f, e);
      }
      for (int t = 0; t < stateCount - 1; t++) {
        to.set(t, f, to.get(t, f) / sum);
      }
      to.set(stateCount - 1, f,1 / sum); // this last operation is needed for caching
    }
    return to;
  }

  private void mySoftmaxGradMx(int c, final Vec betta, Vec state, final Vec nextdS, final Vec prevdS, final Mx dM) {
    VecTools.fill(dM, 0);
    final int stateCount = this.stateCount;
    for (int f = 0; f < stateCount; f++) {
      final double prevS_f = state.get(f);

      double sum = 1;
      {
        for (int i = 0; i < stateCount - 1; i++) {
          sum += MathTools.sqr(getBetta(betta, c, f, i, stateCount, diag));
        }
      }

      double prevdS_f = 0;
      final Mx W = weightMx(betta, c);
      for (int t = 0; t < stateCount; t++) {
        { // dT/dS_prev_f
          prevdS_f += W.get(t, f) * nextdS.get(t);
        }

        { // dT/dM
          final double grad = prevS_f * nextdS.get(t);
          final double betta_ft = t < stateCount - 1 ? getBetta(betta, c, f, t, stateCount, diag) : 0;
          final double betta_ft_2 = MathTools.sqr(betta_ft);
          final double multiply = 2 * grad  / sum / sum;
          for (int j = 0; j < stateCount - 1; j++) {
            final double betta_fj = getBetta(betta, c, f, j, stateCount, diag);
            if (j == t) {
              dM.adjust(f, j, multiply * betta_fj * (sum - MathTools.sqr(betta_fj)));
            }
            else if (t != stateCount - 1) {
              dM.adjust(f, j, -multiply * betta_ft_2 * betta_fj);
            }
            else {
              dM.adjust(f, j, -multiply * betta_fj);
            }
          }
        }
      }
      prevdS.set(f, prevdS_f);
    }
  }

  private static double getBetta(final Vec params, final int c, final int i, final int j, int stateCount, double diag) {
    final int betaSize = 2 * stateCount;
    double value = Math.min(1e9, Math.max(-1e9, params.get(c * betaSize + i) * params.get(c * betaSize + stateCount + j)));
    if (i == j)
      value += diag;
    return value;
  }

  public Vec distribution(Vec betta) {
    Vec[] distribution = new Vec[] {new ArrayVec(stateCount), new ArrayVec(stateCount)};
    VecTools.fill(distribution[0], 1.0 / stateCount);
    //System.out.println("CPU Distribution: " + Arrays.toString(distribution.toArray()));
    for (int i = 0; i < seq.length(); i++) {
      Mx weightMx = weightMx(betta, seq.intAt(i));
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

  public void removeCache() {
    wCache = null;
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


