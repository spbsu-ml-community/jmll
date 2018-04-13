package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
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

    return new PNFAModel(params, stateCount, stateDim, addToDiag, lambda);
  }

  private Vec init(Vec target) {
    final Vec params = new ArrayVec(
        (2 * stateCount - 0) * alphabetSize + stateCount * stateDim
    );
    for (int c = 0; c < alphabetSize; c++) {

      for (int i = 0; i < (2 * stateCount - 0) * alphabetSize; i++) {
        params.set(i, random.nextGaussian());
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

  private static Mx getBetaMx(final Vec params,
                              final int stateCount,
                              final double addToDiag,
                              final int c) {
    final int betaSize = 2 * stateCount - 0;
    final Vec v = params.sub(c * betaSize, stateCount);
    final Vec u = params.sub(c * betaSize + stateCount, stateCount - 0);
    final Mx beta = new VecBasedMx(stateCount, stateCount - 0);
    for (int i = 0; i < stateCount; i++) {
      for (int j = 0; j < stateCount - 0; j++) {
        beta.set(i, j, Math.min(1e9, Math.max(-1e9, v.get(i) * u.get(j))));

      }
    }
    for (int i = 0; i < stateCount - 0; i++) {
      beta.adjust(i, i, addToDiag);
//      beta.set(stateCount - 1, i, 0);
    }


    return beta;
  }

  public static Mx getWeightMx(final Vec params,
                                final int stateCount,
                                final double addToDiag,
                                final double lambda,
                                final int c) {
    final Mx beta = getBetaMx(params, stateCount, addToDiag, c);
    final Mx w = new VecBasedMx(stateCount, stateCount);
    for (int i = 0; i < stateCount; i++) {
      double sum = 0;
      for (int j = 0; j < stateCount - 0; j++) {
        final double e = MathTools.sqr(beta.get(i, j));
        sum += e;
        w.set(i, j, e);
      }
//      w.set(i, stateCount - 1, 1);
//      sum += 1;
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

  private Vec getSeqDistribution(final Vec params, final IntSeq seq) {
    return getSeqDistribution(params, stateCount, addToDiag, lambda, seq);
  }

  private static Vec getSeqDistribution(final Vec params,
                                        final int stateCount,
                                        final double addToDiag,
                                        final double lambda,
                                        final IntSeq seq) {
    Vec distribution = new ArrayVec(stateCount);
    VecTools.fill(distribution, 1.0 / stateCount);
    //System.out.println("CPU Distribution: " + Arrays.toString(distribution.toArray()));
    for (int i = 0; i < seq.length(); i++) {
      Mx weightMx = getWeightMx(params, stateCount, addToDiag, lambda, seq.intAt(i));
      //System.out.println(String.format("-- (%s) CPU WeightMx: %s", i,
      //    Arrays.toString(weightMx.toArray())));
      distribution = multiplyLeft(distribution, weightMx);
    }
    return distribution;
  }

  private Vec getSeqValue(final Vec params, final IntSeq seq) {
    return getSeqValue(params, stateCount, stateDim, addToDiag, lambda, seq);
  }

  private static Vec getSeqValue(final Vec params,
                                 final int stateCount,
                                 final int stateDim,
                                 final double addToDiag,
                                 final double lambda,
                                 final IntSeq seq) {
    return MxTools.multiply(
        getValuesTransposed(params, stateCount, stateDim),
        getSeqDistribution(params, stateCount, addToDiag, lambda, seq)
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
      return (2 * stateCount - 0) * alphabetSize + stateCount * stateDim;
    }

    @Override
    public double value(Vec x) {
      return weight * VecTools.sum2(VecTools.subtract(getSeqValue(x, seq), y));
    }

    @Override
    public Vec gradient(Vec x) {
      final Mx[] betaGrad = new Mx[seqAlphabet.length];
      final Mx[] w = new Mx[seqAlphabet.length];
      final Mx[] beta = new Mx[seqAlphabet.length];
      final Vec[] uv = new Vec[seqAlphabet.length];

      for (int i = 0; i < seqAlphabet.length; i++) {
        betaGrad[i] = new VecBasedMx(stateCount, stateCount - 0);
        w[i] = getWeightMx(x, stateCount, addToDiag, lambda, seqAlphabet[i]);
        beta[i] = getBetaMx(x, stateCount, addToDiag, seqAlphabet[i]);
        uv[i] = x.sub(seqAlphabet[i] * (2 * stateCount - 0), 2 * stateCount - 0);

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
        final Mx outerGrad = new VecBasedMx(stateCount, stateCount);

        for (int from = 0; from < stateCount; from++) {
          for (int to = 0; to < stateCount; to++) {
            for (int valueCol = 0; valueCol < stateDim; valueCol++) {
              outerGrad.adjust(from, to,diff.get(valueCol) * expectedValue.get(to, valueCol));
            }
          }

          VecTools.scale(outerGrad.row(from), 2 * weight * distributions[i].get(from));
        }

        mySoftmaxGradMx(beta[a], w[a], betaGrad[a], outerGrad);
        if (seq.at(seq.length() - 1) == 2 && i == seq.length() - 1) {
          int z = 1;
        }
        expectedValue = MxTools.multiply(w[a], expectedValue);
      }
      final int[] indices = new int[betaGrad.length * (2 * stateCount - 0)];
      final double[] values = new double[indices.length];

      final int uvSize = 2 * stateCount - 0;
      for (int i = 0; i < betaGrad.length; i++) {
        final int betaIdx = uvSize * seqAlphabet[i];
        final Vec v = x.sub(betaIdx, stateCount);
        final Vec u = x.sub(betaIdx + stateCount, stateCount - 0);

        for (int row = 0; row < betaGrad[0].rows(); row++) {
          for (int col = 0; col < betaGrad[0].columns(); col++) {
            final int vIdx = uvSize * i + row;
            values[vIdx] += betaGrad[i].get(row, col) * u.get(col);
            indices[vIdx] = betaIdx + row;

            final int uIdx = uvSize * i + stateCount + col;
            values[uIdx] += betaGrad[i].get(row, col) * v.get(row);
            indices[uIdx] = betaIdx + stateCount + col;

          }
        }

//        for (int j = 0; j < uvSize; j++) {
//          final double val = uv[i].get(j);
//          if (Math.abs(val) > lambda) {
//            if (val > lambda) {
//              values[uvSize * i + j] += lambda;
//            } else {
//              values[uvSize * i + j] -= lambda;
//            }
//          } else {
//            values[uvSize * i + j] = 0;
//            x.set(uvSize * seqAlphabet[i] + j, 0);
//          }
//        }
      }

//      for (int i = 0; i < values.length; i++) {
//        if (values[i] > lambda) values[i] -= lambda;
//        else if (values[i] < -lambda) values[i] += lambda;
//        else values[i] = 0;
//      }

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
      Vec seqDistribution = getSeqDistribution(params, seq);
      final Vec grad = VecTools.subtract(
          MxTools.multiply(getValuesTransposed(params), seqDistribution), y
      );
      final double[] result = new double[stateCount * stateDim];
      final int[] indices = new int[stateCount * stateDim];
      for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateDim; j++) {
          final int idx = i * stateDim + j;
          result[idx] = 2 * weight * grad.get(j) * seqDistribution.get(i);
          indices[idx] = (2 * stateCount - 0) * alphabetSize + idx;
        }
      }

      return new SparseVec(dim(), indices, result);
    }

    @Override
    public double value(Vec params) {
      return weight * VecTools.sum2(VecTools.subtract(getSeqValue(params, seq), y));
    }

    @Override
    public int dim() {
      return (2 * stateCount - 0) * alphabetSize + stateCount * stateDim;
    }
  }

  static class PNFAModel implements Function<Seq<Integer>, Vec> {
    private ArrayVec params;
    private int stateCount;
    private int stateDim;
    private double addToDiag;
    private double lambda;

    public PNFAModel(Vec params, int stateCount, int stateDim, double addToDiag, double lambda) {
      this.params = new ArrayVec(params.toArray());
      this.stateCount = stateCount;
      this.stateDim = stateDim;
      this.addToDiag = addToDiag;
      this.lambda = lambda;
    }

    @Override
    public Vec apply(Seq<Integer> seq) {
      return getSeqValue(params, stateCount, stateDim, addToDiag, lambda, (IntSeq) seq);
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

    public double getAddToDiag() {
      return addToDiag;
    }

    public void setAddToDiag(double addToDiag) {
      this.addToDiag = addToDiag;
    }

    public double getLambda() {
      return lambda;
    }

    public void setLambda(double lambda) {
      this.lambda = lambda;
    }
  }


  // gradient of softmax multiplied by derivative of some outer function by each row
  private static void mySoftmaxGradMx(final Mx beta,
                                      final Mx softmaxValues,
                                      final Mx gradTo,
                                      final Mx outerGrad) {
    final int rows = softmaxValues.rows();
    final int columns = softmaxValues.columns();
    for (int from = 0; from < rows; from++) {
      double sum = 1;
      for (int j = 0; j < columns - 0; j++) {
        sum += MathTools.sqr(beta.get(from, j));
      }

      for (int to = 0; to < columns; to++) {
        final double curW = softmaxValues.get(from, to);
        final double grad = outerGrad.get(from, to);
        for (int j = 0; j < columns - 0; j++) {
          if (j == to) {
            gradTo.adjust(from, j, 2 * grad * beta.get(from, j) * (sum - MathTools.sqr(beta.get(from, j))) / sum / sum);
          }
          else if (to != columns - 0) {
            gradTo.adjust(from, j, -2 * grad * MathTools.sqr(beta.get(from, to)) * beta.get(from, j) / sum / sum);
          } else {
            gradTo.adjust(from, j, -2 * grad * beta.get(from, j) / sum / sum);
          }
        }
      }
    }
  }
}
