package com.expleague.cuda.root.nn;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.Random;

import static com.expleague.commons.math.vectors.VecTools.incscale;

/**
 * Created by hrundelb on 23.08.17.
 */
public class PNFAParamsGPU<T> {
  private Vec values;
  private Mx[] w;
  private Mx[] wTrans;

  private final Alphabet<T> alphabet;
  private final int stateCount;

  private Mx[] beta;
  private double lambda = -0.00001;

  public PNFAParamsGPU(final Random random, final int stateCount, final Alphabet<T> alphabet) {
    this.stateCount = stateCount;
    this.alphabet = alphabet;

    beta = new Mx[alphabet.size()];
    values = new ArrayVec(stateCount);
    w = new Mx[alphabet.size()];
    wTrans = new Mx[alphabet.size()];

    for (int i = 0; i < alphabet.size(); i++) {
      w[i] = new VecBasedMx(stateCount, stateCount);
      wTrans[i] = new VecBasedMx(stateCount, stateCount);
      beta[i] = new VecBasedMx(stateCount, stateCount - 1);
    }

    for (int i = 0; i < stateCount; i++) {
      values.set(i, random.nextGaussian());
    }

    for (int a = 0; a < alphabet.size(); a++) {
      for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateCount - 1; j++) {
          beta[a].set(i, j, random.nextGaussian());
        }
        if (i < stateCount - 1)
          beta[a].set(i, i, 3);
      }
    }
    updateWeights();
  }

  public PNFAParamsGPU.PNFAParamsGradGPU calcSeqGrad(final Seq<T> seq, final int[] seqAlphabet,
                                                     final double targetVal) {
    final double diff = getSeqValue(seq) - targetVal;
    final TIntIntHashMap alphabetMap = new TIntIntHashMap(seqAlphabet.length);
    for (int i = 0; i < seqAlphabet.length; i++) {
      alphabetMap.put(seqAlphabet[i], i);
    }

    final Mx[] betaGrad = new Mx[seqAlphabet.length];
    for (int i = 0; i < seqAlphabet.length; i++) {
      betaGrad[i] = new VecBasedMx(stateCount, stateCount - 1);
    }

    final Vec[] distributions = new Vec[seq.length() + 1];
    distributions[0] = new ArrayVec(stateCount);
    distributions[0].set(0, 1);

    for (int i = 0; i < seq.length(); i++) {
      distributions[i + 1] = MxTools.multiply(wTrans[alphabet.index(seq.at(i))], distributions[i]);
    }

    Vec expectedValue = new ArrayVec(stateCount);
    Vec curDistr = new ArrayVec(stateCount);

    for (int i = 0; i < stateCount; i++) {
      expectedValue.set(i, values.get(i));
    }

    for (int i = seq.length() - 1; i >= 0; i--) {
      final int c = alphabet.index(seq.at(i));
      final int a = alphabetMap.get(c);

      for (int to = 0; to < stateCount; to++) {
        VecTools.fill(curDistr, 0);
        curDistr.set(to, 1);
        for (int from = 0; from < stateCount; from++) {
          for (int j = 0; j < stateCount - 1; j++) {
            final double grad = 2 * diff * distributions[i].get(from) * expectedValue.get(to);
            final double curW = w[c].get(from, to);
            if (j == to) {
              betaGrad[a].adjust(from, j, grad * curW * (1 - curW));
            } else {
              betaGrad[a].adjust(from, j, -grad * curW * w[c].get(from, j));
            }
          }
        }
      }
      expectedValue = MxTools.multiply(w[c], expectedValue);
    }

    final Vec valuesGrad = new ArrayVec(stateCount);
    for (int i = 0 ; i < stateCount; i++) {
      valuesGrad.set(i, 2 * diff * distributions[seq.length()].at(i));
    }

    for (int t = 0; t < betaGrad.length; t++) {
      Mx mxBeta = betaGrad[t];
      for (int i = 0; i < mxBeta.dim(); i++) {
        mxBeta.adjust(i, 2 * lambda * beta[seqAlphabet[t]].get(i));
      }
    }

    return new PNFAParamsGPU.PNFAParamsGradGPU(betaGrad, valuesGrad);
  }

  public double getSeqValue(final Seq<T> seq) {
    Vec distrib = new ArrayVec(stateCount);
    distrib.set(0, 1);
    for (int s = 0; s < seq.length(); s++) {
      distrib = MxTools.multiply(wTrans[alphabet.index(seq.at(s))], distrib);
    }
    return VecTools.multiply(distrib, values);
  }

  public void updateParams(final Mx[] addBeta, final Vec addValues, final double step, final int[] alpha) {
    incscale(values, addValues, step);
    final int alphabetSize = alpha == null ? alphabet.size() : alpha.length;

    for (int charId = 0; charId < alphabetSize; charId++) {
      final int c = alpha == null ? charId : alpha[charId];
      for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateCount - 1; j++) {
          beta[c].adjust(i, j, addBeta[charId].get(i, j) * step);
        }
      }
    }

    updateWeights(alpha);
  }

  public void setParams(final PNFAParamsGPU<T> params) {
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

  public void updateWeights() {
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
          final double val = Math.exp(beta[c].get(i, j)) / (1 + sum);
          w[c].set(i, j, val);
          wTrans[c].set(j, i, val);
        }

        w[c].set(i, stateCount - 1, 1 / (1 + sum));
        wTrans[c].set(stateCount - 1, i, 1 / (1 + sum));
      }
    }
  }

  public static class PNFAParamsGradGPU {
    private Mx[] betaGrad;
    private Vec valuesGrad;

    PNFAParamsGradGPU(final Mx[] betaGrad, final Vec valuesGrad) {
      this.betaGrad = betaGrad;
      this.valuesGrad = valuesGrad;
    }

    public Mx[] getBetaGrad() {
      return betaGrad;
    }

    public Vec getValuesGrad() {
      return valuesGrad;
    }
  }
}