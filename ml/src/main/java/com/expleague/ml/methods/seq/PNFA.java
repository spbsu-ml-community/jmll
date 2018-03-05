package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.optimization.Optimize;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import org.apache.commons.math3.util.FastMath;

import java.util.*;
import java.util.function.Function;

public class PNFA<Loss extends WeightedL2> implements SeqOptimization<Integer, Loss> {
  private final int stateCount;
  private final int stateDim;
  private final int alphabetSize;
  private final Random random;
  private final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize, valuesOptimize;
  private final double lambda;
  private final double addToDiag;
  private final int weightValueIterCount;

  public PNFA(
      final int stateCount,
      final int stateDim,
      final int alphabetSize,
      final double lambda,
      final double addToDiag,
      final Random random,
      final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize,
      final Optimize<FuncEnsemble<? extends FuncC1>> valuesOptimize,
      final int weightValueIterCount
  ) {
    this.stateCount = stateCount;
    this.stateDim = stateDim;
    this.alphabetSize = alphabetSize;
    this.lambda = lambda;
    this.addToDiag = addToDiag;
    this.random = random;
    this.weightsOptimize= weightsOptimize;
    this.valuesOptimize = valuesOptimize;
    this.weightValueIterCount = weightValueIterCount;
  }

  @Override
  public Function<Seq<Integer>, Vec> fit(final DataSet<Seq<Integer>> learn, final Loss loss) {
    Vec params = init(loss.target());
    FuncC1[] funcs = new FuncC1[learn.length()];
    for (int iter = 0; iter < weightValueIterCount; iter++) {
      for (int i = 0; i < learn.length(); i++) {
        final IntSeq seq = (IntSeq) learn.at(i);
        funcs[i] = new PNFAPointLossFunc(
            seq, loss.target().sub(i * stateDim, stateDim), loss.getWeights().get(i)
        );
      }

      params = weightsOptimize.optimize(new FuncEnsemble<>(Arrays.asList(funcs), 1), params);

      for (int i = 0; i < learn.length(); i++) {
        final IntSeq seq = (IntSeq) learn.at(i);
        funcs[i] = new PNFAPointValueLossFunc(
            seq, loss.target().sub(i* stateDim, stateDim), loss.getWeights().get(i)
        );
      }

      System.out.println("Values before: " + getValues(params));
      params = valuesOptimize.optimize(new FuncEnsemble<>(Arrays.asList(funcs), 1), params);
      System.out.println("Values after: " + getValues(params));
    }

    return new PNFAModel(params, stateCount, stateDim);
    // (seq) -> getSeqValue(optParams, (IntSeq) seq);
  }

  private Vec init(Vec target) {
    final Vec params = new ArrayVec(
        stateCount * (stateCount - 1) * alphabetSize + stateCount * stateDim
    );
    for (int c = 0; c < alphabetSize; c++) {
      final Mx beta = getMx(params, c);

      for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateCount - 1; j++) {
          beta.set(i, j, random.nextGaussian());
        }
        VecTools.normalizeL2(beta.row(i));
      }
      for (int i = 0; i < stateCount - 1; i++) {
        beta.adjust(stateCount - 1, i, -addToDiag);
        beta.adjust(i, i, addToDiag);
      }
    }

    final Mx values = getValues(params);
    final Mx targetValuesMx = new VecBasedMx(stateDim, target);
    for (int col = 0; col < targetValuesMx.columns(); col++) {
      final double[] targetValues = targetValuesMx.col(col).toArray();
      Arrays.sort(targetValues);
      List<Double> medians = new ArrayList<>(stateCount);
      for (int i = 0; i < stateCount; i++) {
        medians.add(targetValues[(int) ((i + 0.5) * target.dim() / stateDim / stateCount)]);
      }
      Collections.shuffle(medians, random);
      for (int i = 0; i < stateCount; i++) {
        values.set(i, col, medians.get(i));
      }
    }

    return params;
  }

  private Mx getMx(final Vec params, final int c) {
    return getMx(params, stateCount, c);
  }

  private static Mx getMx(final Vec params,
                          final int stateCount,
                          final int c) {
    final int mxSize = stateCount * (stateCount - 1);
    return new VecBasedMx(stateCount - 1, params.sub(c * mxSize, mxSize));
  }

  private static Mx getWeightMx(final Vec params,
                                final int stateCount,
                                final int c) {
    final Mx beta = getMx(params, stateCount, c);
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

  private Mx getValues(final Vec params) {
    return getValues(params, stateCount, stateDim);
  }

  private static Mx getValues(final Vec params, final int stateCount, final int stateDim) {
    return new VecBasedMx(
        stateDim,
        params.sub(params.dim() - stateCount * stateDim, stateCount * stateDim)
    );
  }

  private Mx getValuesTransposed(final Vec params) {
    return getValuesTransposed(params, stateCount, stateDim);
  }

  private static Mx getValuesTransposed(final Vec params,
                                        final int stateCount,
                                        final int stateDim) {
    return MxTools.transpose(getValues(params, stateCount, stateDim));
  }

  private static Vec getSeqDistribution(final Vec params,
                                        final int stateCount,
                                        final IntSeq seq) {
    Vec distribution = new ArrayVec(stateCount);
    VecTools.fill(distribution, 1.0 / stateCount);
    //System.out.println("CPU Distribution: " + Arrays.toString(distribution.toArray()));
    for (int i = 0; i < seq.length(); i++) {
      Mx weightMx = getWeightMx(params, stateCount, seq.intAt(i));
      //System.out.println(String.format("-- (%s) CPU WeightMx: %s", i,
      //    Arrays.toString(weightMx.toArray())));
      distribution = multiplyLeft(distribution, weightMx);
    }
    return distribution;
  }

  private static Vec getSeqValue(final Vec params,
                                 final int stateCount,
                                 final int stateDim,
                                 final IntSeq seq) {
    return MxTools.multiply(
        getValuesTransposed(params, stateCount, stateDim),
        getSeqDistribution(params, stateCount, seq)
    );
  }

  public class PNFAPointLossFunc extends FuncC1.Stub {

    private final IntSeq seq;
    private final Vec y;
    private final double weight;
    private final int[] seqAlphabet;
    private final TIntIntMap alphabetToOrderMap = new TIntIntHashMap();

    public PNFAPointLossFunc(final IntSeq seq, final Vec y, final double weight) {
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
      return stateCount * (stateCount - 1) * alphabetSize + stateCount * stateDim;
    }

    @Override
    public double value(Vec x) {
      return weight * VecTools.sum2(VecTools.subtract(getSeqValue(x, stateCount, stateDim, seq), y));
    }

    @Override
    public Vec gradient(Vec x) {
      final Mx[] betaGrad = new Mx[seqAlphabet.length];
      final Mx[] w = new Mx[seqAlphabet.length];

      for (int i = 0; i < seqAlphabet.length; i++) {
        betaGrad[i] = new VecBasedMx(stateCount, stateCount - 1);
        w[i] = getWeightMx(x, stateCount, seqAlphabet[i]);
      }

      final Vec[] distributions = new Vec[seq.length() + 1];

      distributions[0] = new ArrayVec(stateCount);
      VecTools.fill(distributions[0], 1.0 / stateCount);

      for (int i = 0; i < seq.length(); i++) {
        distributions[i + 1] = multiplyLeft(distributions[i], w[alphabetToOrderMap.get(seq.intAt(i))]);
      }

      Mx expectedValue = getValues(x);

      final Vec diff = VecTools.subtract(
          MxTools.multiply(getValuesTransposed(x), distributions[seq.length()]), y
      );

      for (int i = seq.length() - 1; i >= 0; i--) {
        final int a = alphabetToOrderMap.get(seq.intAt(i));


        for (int to = 0; to < stateCount; to++) {
          for (int from = 0; from < stateCount; from++) {
            for (int j = 0; j < stateCount - 1; j++) {
              final double curW = w[a].get(from, to);
              double grad = 0;
              for (int valueCol = 0; valueCol < stateDim; valueCol++) {
                grad += diff.get(valueCol) * expectedValue.get(to, valueCol);
              }
              grad = grad * 2 * weight * distributions[i].get(from) + lambda * curW;
              //                double grad = 2 * weight * diff.get(valueCol) * distributions[i].get(from) *
              //                    expectedValue.get(to, valueCol) + lambda * curW;

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

  private static Vec multiplyLeft(Vec vec, Mx mx) {
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
    private final Vec y;
    private final double weight;

    public PNFAPointValueLossFunc(IntSeq seq, Vec y, double weight) {
      this.seq = seq;
      this.y = y;
      this.weight = weight;
    }

    @Override
    public Vec gradient(Vec params) {
      Vec seqDistribution = getSeqDistribution(params, stateCount, seq);
      final Vec grad = VecTools.subtract(
          MxTools.multiply(getValuesTransposed(params), seqDistribution), y
      );
      final double[] result = new double[stateCount * stateDim];
      final int[] indices = new int[stateCount * stateDim];
      for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateDim; j++) {
          final int idx = i * stateDim + j;
          result[idx] = 2 * weight * grad.get(j) * seqDistribution.get(i);
          indices[idx] = stateCount * (stateCount - 1) * alphabetSize + idx;
        }
      }
      return new SparseVec(dim(), indices, result);
    }

    @Override
    public double value(Vec params) {
      return weight * VecTools.sum2(VecTools.subtract(getSeqValue(params, stateCount, stateDim, seq), y));
    }

    @Override
    public int dim() {
      return stateCount * (stateCount - 1) * alphabetSize + stateCount * stateDim;
    }
  }

  static class PNFAModel implements Function<Seq<Integer>, Vec> {
    private ArrayVec params;
    private int stateCount;
    private int stateDim;

    public PNFAModel(Vec params, int stateCount, int stateDim) {
      this.params = new ArrayVec(params.toArray());
      this.stateCount = stateCount;
      this.stateDim = stateDim;
    }

    @Override
    public Vec apply(Seq<Integer> seq) {
      return getSeqValue(params, stateCount, stateDim, (IntSeq) seq);
    }

    public ArrayVec getParams() {
      return params;
    }

    public void setParams(ArrayVec params) {
      this.params = params;
    }

    public int getStateCount() {
      return stateCount;
    }

    public void setStateCount(int stateCount) {
      this.stateCount = stateCount;
    }

    public int getStateDim() {
      return stateDim;
    }

    public void setStateDim(int stateDim) {
      this.stateDim = stateDim;
    }
  }
}
