package com.spbsu.ml.methods.seq;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.WeightedL2;
import com.spbsu.ml.methods.SeqOptimization;
import com.spbsu.ml.optimization.Optimize;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.Random;

public class PNFA<Loss extends WeightedL2> implements SeqOptimization<Integer, Loss> {
  private final int stateCount;
  private final int alphabetSize;
  private final Random random;
  private final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize, valuesOptimize;
  private static final double lambda = -0.005;
  private final int weightValueIterCount;

  public PNFA(final int stateCount, final int alphabetSize, final Random random,
              final Optimize<FuncEnsemble<? extends FuncC1>> optimize, final int weightValueIterCount) {
    this(stateCount, alphabetSize, random, optimize, optimize, weightValueIterCount);
  }

  public PNFA(final int stateCount, final int alphabetSize, final Random random,
              final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize,
              final Optimize<FuncEnsemble<? extends FuncC1>> valuesOptimize, final int weightValueIterCount) {
    this.stateCount = stateCount;
    this.alphabetSize = alphabetSize;
    this.random = random;
    this.weightsOptimize= weightsOptimize;
    this.valuesOptimize = valuesOptimize;
    this.weightValueIterCount = weightValueIterCount;
  }

  @Override
  public Computable<Seq<Integer>, Vec> fit(final DataSet<Seq<Integer>> learn, final Loss loss) {
    Vec params = init(loss.target());
    FuncC1[] funcs = new FuncC1[learn.length()];
    for (int iter = 0; iter < weightValueIterCount; iter++) {
      for (int i = 0; i < learn.length(); i++) {
        final IntSeq seq = (IntSeq) learn.at(i);
        funcs[i] = new PNFAPointLossFunc(seq, loss.target().get(i), loss.getWeights().get(i));
      }

      params = weightsOptimize.optimize(new FuncEnsemble<>(Arrays.asList(funcs), 1), params);

      for (int i = 0; i < learn.length(); i++) {
        final IntSeq seq = (IntSeq) learn.at(i);
        funcs[i] = new PNFAPointValueLossFunc(seq, loss.target().get(i), loss.getWeights().get(i));
      }

      System.out.println("Values before: " + getValues(params));
      params = valuesOptimize.optimize(new FuncEnsemble<>(Arrays.asList(funcs), 1), params);
      System.out.println("Values after: " + getValues(params));
    }

    final Vec optParams = params;
    return (seq) -> new SingleValueVec(getSeqValue(optParams, (IntSeq) seq));
  }

  private Vec init(Vec target) {
    final Vec params = new ArrayVec(stateCount * (stateCount - 1) * alphabetSize + stateCount);
    for (int c = 0; c < alphabetSize; c++) {
      final Mx beta = getMx(params, c);
      for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateCount - 1; j++) {
          beta.set(i, j, random.nextGaussian());
        }
        if (i < stateCount - 1) {
          beta.adjust(i, i, stateCount / 2.0 + 3); // TODO change it
        }
      }

      for (int i = 0; i < stateCount - 1; i++) {
        beta.adjust(stateCount - 1, i, -stateCount / 2.0 - 3); // TODO change it
      }
    }

    final Vec values = getValues(params);
    final double[] targetValues = target.toArray();
    Arrays.sort(targetValues);
    for (int i = 0; i < stateCount; i++) {
      values.set(i, targetValues[(int) ((i + 0.5) * target.dim() / stateCount)]);
    }

    return params;
  }

  private Mx getMx(final Vec params, final int c) {
    final int mxSize = stateCount * (stateCount - 1);
    return new VecBasedMx(stateCount - 1, params.sub(c * mxSize, mxSize));
  }

  private Mx getWeightMx(final Vec params, final int c) {
    final Mx beta = getMx(params, c);
    final Mx w = new VecBasedMx(stateCount, stateCount);
    for (int i = 0; i < stateCount; i++) {
      double sum = 0;
      for (int j = 0; j < stateCount - 1; j++) {
        final double e = FastMath.exp(beta.get(i, j));
        sum += e;
        w.set(i, j, e);
      }
      w.set(i, stateCount - 1, 1);
      sum += 1;
      for (int j = 0; j < stateCount; j++) {
        w.set(i, j, w.get(i, j) / sum);
      }
    }

    return w;
  }

  private Vec getValues(final Vec params) {
    return params.sub(params.dim() - stateCount, stateCount);
  }

  private Vec getSeqDistribution(final Vec params, final IntSeq seq) {
    Vec distribution = new ArrayVec(stateCount);
    VecTools.fill(distribution, 1.0 / stateCount);
    for (int i = 0; i < seq.length(); i++) {
      distribution = multiplyLeft(distribution, getWeightMx(params, seq.intAt(i)));
    }
    return distribution;
  }

  private double getSeqValue(final Vec params, final IntSeq seq) {
    return VecTools.multiply(getSeqDistribution(params, seq), getValues(params));
  }

  private class PNFAPointLossFunc extends FuncC1.Stub {

    private final IntSeq seq;
    private final double y;
    private final double weight;
    private final int[] seqAlphabet;
    private final TIntIntMap alphabetToOrderMap = new TIntIntHashMap();

    public PNFAPointLossFunc(final IntSeq seq, final double y, final double weight) {
      this.seq = seq;
      this.y = y;
      this.weight = weight;

      this.seqAlphabet = seq.stream().sorted().distinct().toArray();
      for (int i = 0; i < seqAlphabet.length; i++) {
        alphabetToOrderMap.put(seqAlphabet[i], i);
      }
    }

    @Override
    public int dim() {
      return stateCount * (stateCount - 1) * alphabetSize + stateCount;
    }

    @Override
    public double value(Vec x) {
      return weight * MathTools.sqr(getSeqValue(x, seq) - y);
    }

    @Override
    public Vec gradient(Vec x) {
      final Mx[] betaGrad = new Mx[seqAlphabet.length];
      final Mx[] w = new Mx[seqAlphabet.length];

      for (int i = 0; i < seqAlphabet.length; i++) {
        betaGrad[i] = new VecBasedMx(stateCount, stateCount - 1);
        w[i] = getWeightMx(x, seqAlphabet[i]);
      }

      final Vec[] distributions = new Vec[seq.length() + 1];

      distributions[0] = new ArrayVec(stateCount);
      VecTools.fill(distributions[0], 1.0 / stateCount);

      for (int i = 0; i < seq.length(); i++) {
        distributions[i + 1] = multiplyLeft(distributions[i], w[alphabetToOrderMap.get(seq.intAt(i))]);
      }

      Vec expectedValue = new ArrayVec(stateCount);

      for (int i = 0; i < stateCount; i++) {
        expectedValue.set(i, getValues(x).get(i));
      }

      final double diff = VecTools.multiply(distributions[seq.length()], getValues(x)) - y;

      for (int i = seq.length() - 1; i >= 0; i--) {
        final int a = alphabetToOrderMap.get(seq.intAt(i));

        for (int to = 0; to < stateCount; to++) {
          for (int from = 0; from < stateCount; from++) {
            for (int j = 0; j < stateCount - 1; j++) {
              final double curW = w[a].get(from, to);
              final double grad = 2 * weight * diff * distributions[i].get(from) * expectedValue.get(to);
              if (j == to) {
                betaGrad[a].adjust(from, j, grad * curW * (1 - curW));
              }
              else {
                betaGrad[a].adjust(from, j, -grad * curW * w[a].get(from, j));
              }
            }
          }
        }
        expectedValue = MxTools.multiply(w[a], expectedValue);
      }

      for (int i = 0; i < seqAlphabet.length; i++) {
        for (int to = 0; to < stateCount; to++) {
          for (int from = 0; from < stateCount; from++) {
            for (int j = 0; j < stateCount - 1; j++) {
              final double curW = w[i].get(from, to);
              final double grad = lambda * w[i].get(from, to);
              if (j == to) {
                betaGrad[i].adjust(from, j, grad * curW * (1 - curW));
              }
              else {
                betaGrad[i].adjust(from, j, -grad * curW * w[i].get(from, j));
              }
            }
          }
        }
      }

      final int[] indices = new int[betaGrad.length * betaGrad[0].dim()];
      final double[] values = new double[indices.length];
      for (int i = 0; i < betaGrad.length; i++) {
        for (int j = 0; j < betaGrad[0].dim(); j++) {
          final int index = i * betaGrad[0].dim() + j;
          indices[index] = seqAlphabet[i] * betaGrad[0].dim() + j;
          values[index] = betaGrad[i].get(j);
        }
      }
      return new SparseVec(dim(), indices, values);
    }
  }

  private Vec multiplyLeft(Vec vec, Mx mx) {
    final Vec result = new ArrayVec(vec.dim());
    for (int i = 0; i < mx.columns(); i++) {
      double x = 0;
      for (int j = 0; j < mx.rows(); j++) {
        x += vec.get(j) * mx.get(j, i);
      }
      result.set(i, x);
    }
    return result;
  }

  private class PNFAPointValueLossFunc extends FuncC1.Stub {
    private final IntSeq seq;
    private final double y;
    private final double weight;

    public PNFAPointValueLossFunc(IntSeq seq, double y, double weight) {
      this.seq = seq;
      this.y = y;
      this.weight = weight;
    }

    @Override
    public Vec gradient(Vec params) {
      Vec seqDistribution = getSeqDistribution(params, seq);
      final double grad = VecTools.multiply(seqDistribution, getValues(params)) - y;
      final double[] result = new double[stateCount];
      final int[] indices = new int[stateCount];
      for (int i = 0; i < stateCount; i++) {
        result[i] = 2 * grad * seqDistribution.get(i);
        indices[i] = stateCount * (stateCount - 1) * alphabetSize + i;
      }
      return new SparseVec(dim(), indices, result);
    }

    @Override
    public double value(Vec params) {
      return MathTools.sqr(VecTools.multiply(getSeqDistribution(params, seq), getValues(params)) - y);
    }

    @Override
    public int dim() {
      return stateCount * (stateCount - 1) * alphabetSize + stateCount;    }
  }
}
