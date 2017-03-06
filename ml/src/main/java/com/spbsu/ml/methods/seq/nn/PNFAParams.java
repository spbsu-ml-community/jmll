package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.List;
import java.util.Random;

public class PNFAParams<T> {
  private Vec values; //todo getter and private
  private Mx[] w;

  private final Alphabet<T> alphabet;
  private final int stateCount;

  private final Mx[] wGrad;
  private final Vec valuesGrad;

  private Mx[] beta;

  public PNFAParams(final Random random, final int stateCount, final Alphabet<T> alphabet) {
    this.stateCount = stateCount;
    this.alphabet = alphabet;

    beta = new Mx[alphabet.size()];
    values = new ArrayVec(stateCount);
    w = new Mx[alphabet.size()];

    wGrad = new Mx[alphabet.size()];
    valuesGrad = new ArrayVec(stateCount);

    for (int i = 0; i < alphabet.size(); i++) {
      w[i] = new VecBasedMx(stateCount, stateCount);
      beta[i] = new VecBasedMx(stateCount, stateCount - 1);

      wGrad[i] = new VecBasedMx(stateCount, stateCount);
    }

    for (int i = 0; i < stateCount; i++) {
      values.set(i, 1);
    }

//    for (int a = 0; a < alphabet.size(); a++) {
//      for (int i = 0; i < stateCount; i++) {
//        for (int j = 0; j < stateCount - 1; j++) {
//          beta[a].set(i, j, random.nextDouble() - 0.5);
//        }
//      }
//    }
    updateWeights();
  }

  public PNFAParamsGrad calcSeqGrad(final Seq<T> seq, final int[] seqAlphabet, final double targetVal) {
    final double diff = getSeqValue(seq) - targetVal;
    final TIntIntHashMap alphabetMap = new TIntIntHashMap(seqAlphabet.length);
    for (int i = 0; i < seqAlphabet.length; i++) {
      alphabetMap.put(seqAlphabet[i], i);
    }

    final Mx[] wGrad = new Mx[seqAlphabet.length];
    for (int i = 0; i < seqAlphabet.length; i++) {
      wGrad[i] = new VecBasedMx(stateCount, stateCount);
    }
    final Vec valuesGrad = new ArrayVec(stateCount);

    final Vec[] distributions = new Vec[seq.length() + 1];

    for (int i = 0; i <= seq.length(); i++) {
      distributions[i] = new ArrayVec(stateCount);
    }
    distributions[0].set(0, 1);

    for (int i = 0; i < seq.length(); i++) {
      mulLeftTo(distributions[i], w[alphabet.index(seq.at(i))], distributions[i + 1]);
    }

    Mx rightProd = new VecBasedMx(stateCount, stateCount), tmpProd = new VecBasedMx(stateCount, stateCount);
    Vec curDistr = new ArrayVec(stateCount), rightDistribution = new ArrayVec(stateCount);

    for (int i = 0; i < stateCount; i++) {
      rightProd.set(i, i, values.get(i));
    }

    for (int i = seq.length() - 1; i >= 0; i--) {
      final int a = alphabetMap.get(alphabet.index(seq.at(i)));

      for (int to = 0; to < stateCount; to++) {
        VecTools.fill(curDistr, 0);
        curDistr.set(to, 1);
        mulLeftTo(curDistr, rightProd, rightDistribution);
        final double sum = VecTools.sum(rightDistribution);
        for (int from = 0; from < stateCount; from++) {
          wGrad[a].adjust(from, to, 2 * diff * distributions[i].get(from) * sum);
        }
      }
      VecTools.fill(tmpProd, 0);
      MxTools.multiplyTo(w[a], rightProd, tmpProd);
      Mx tmp = tmpProd;
      tmpProd = rightProd;
      rightProd = tmp;

    }

    for (int i = 0 ; i < stateCount; i++) {
      valuesGrad.set(i, 2 * diff * distributions[seq.length()].at(i));
    }

    return new PNFAParamsGrad(wGrad, valuesGrad);
  }

  public PNFAParamsGrad calcPathGrad(final Seq<Seq<T>> learn, final Vec target, final int[] path) {
    for (int a = 0; a < alphabet.size(); a++) {
      VecTools.fill(wGrad[a], 0);
    }
    VecTools.fill(valuesGrad, 0);

    for (int seqIndex = 0; seqIndex < learn.length(); seqIndex++) {
      final Seq<T> seq = learn.at(seqIndex);
      final double diff = getSeqValue(seq) - target.at(seqIndex);
      double seqPathProbab = 1;
      for (int at = 0; at < seq.length(); at++) {
        final int a = alphabet.index(seq.at(at));
        seqPathProbab *= w[a].get(path[at], path[at + 1]);
      }

      final double lastValue = values.at(path[seq.length()]);
      for (int at = 0; at < seq.length(); at++) {
        final int a = alphabet.index(seq.at(at));
        final double curW = w[a].get(path[at], path[at + 1]);
        if (Math.abs(curW) < MathTools.EPSILON) {
          continue;
        }
        wGrad[a].adjust(path[at], path[at + 1], 2 * diff * lastValue * seqPathProbab / curW);
      }

      valuesGrad.adjust(path[seq.length()], 2 * diff * seqPathProbab);
    }

    return new PNFAParamsGrad(wGrad, valuesGrad);
  }

  public double getSeqValue(final Seq<T> seq) {
    Mx distribution = new VecBasedMx(1, stateCount), distribution1 = new VecBasedMx(1, stateCount);
    distribution.set(0, 1);
    final int length = seq.length();
    for (int j = 0; j < length; j++) {
      VecTools.fill(distribution1, 0);
      MxTools.multiplyTo(distribution, w[alphabet.index(seq.at(j))], distribution1);
      final Mx tmp = distribution;
      distribution = distribution1;
      distribution1 = tmp;
    }
    return VecTools.multiply(distribution, values);
  }

  public void updateParams(final Mx[] addW, final Vec addValues, final double step) {
    updateParams(addW, addValues, step, null);
  }

  public void updateParams(final Mx[] addW, final Vec addValues, final double step, final int[] alpha) {
    VecTools.incscale(values, addValues, step);

    final int alphabetSize = alpha == null ? alphabet.size() : alpha.length;

    for (int charId = 0; charId < alphabetSize; charId++) {
      final int c = alpha == null ? charId : alpha[charId];
      for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateCount - 1; j++) {
          for (int k = 0; k < stateCount; k++) {
            if (j == k) {
              beta[c].adjust(i, j, addW[charId].get(i, k) * w[c].get(i, k) * (1 - w[c].get(i, k)) * step);
            } else {
              beta[c].adjust(i, j, addW[charId].get(i, k) * -w[c].get(i, k) * w[c].get(i, j) * step);
            }
          }
        }
      }
    }

    updateWeights(alpha);
  }

  public Vec getNextStateDistribution(final int curState, final int posInSeq, final List<Seq<T>> learn,
                                      final int startSeqIndex, final int endSeqIndex) {
    final Vec distribution = new ArrayVec(stateCount);
    for (int i = startSeqIndex; i < endSeqIndex; i++) {
      final Seq<T> seq = learn.get(i);
      if (posInSeq < seq.length()) {
        for (int j = 0; j < stateCount; j++) {
          distribution.adjust(j, w[alphabet.index(seq.at(posInSeq))].get(curState, j));
        }
      }
    }
    return distribution;
  }

  public void setParams(final PNFAParams<T> params) {
    this.beta = params.beta;
    this.w = params.w;
    this.values = params.values;
  }

  public void setValues(Vec values) {
    this.values = values;
  }

  public Mx[] getW() {
    return w;
  }

  public void setW(Mx[] w) {
    this.w = w;
  }

  public Mx[] getBeta() {
    return beta;
  }

  public Vec getValues() {
    return values;
  }

  private void updateWeights() {
    updateWeights(null);
  }

  private void updateWeights(final int[] alpha) {
    final int alphabetSize = alpha == null ? alphabet.size() : alpha.length;

    for (int charId = 0; charId < alphabetSize; charId++) {
      final int c = alpha == null ? charId : alpha[charId];

      for (int i = 0; i < stateCount; i++) {
        double sum = 0;
        for (int j = 0; j < stateCount - 1; j++) {
          sum += Math.exp(beta[c].get(i, j));
        }
        for (int j = 0; j < stateCount - 1; j++) {
          w[c].set(i, j, Math.exp(beta[c].get(i, j)) / (1 + sum));
        }
        w[c].set(i, stateCount - 1, 1 / (1 + sum));
      }
    }
  }

  private void mulLeftTo(final Vec vec, final Mx mx, final Vec dest) {
    for (int i = 0; i < vec.length(); i++) {
      dest.set(i, 0);
      for (int j = 0; j < mx.rows(); j++) {
        dest.adjust(i, vec.get(j) * mx.get(j, i));
      }
    }
  }

  public static class PNFAParamsGrad {
    private Mx[] wGrad;
    private Vec valuesGrad;

    PNFAParamsGrad(final Mx[] wGrad, final Vec valuesGrad) {
      this.wGrad = wGrad;
      this.valuesGrad = valuesGrad;
    }

    public Mx[] getWGrad() {
      return wGrad;
    }

    public Vec getValuesGrad() {
      return valuesGrad;
    }
  }
}

