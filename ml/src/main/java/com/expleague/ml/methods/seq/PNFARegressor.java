package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.*;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.optimization.Optimize;

import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;

public class PNFARegressor<Type, Loss extends WeightedL2> implements SeqOptimization<Type, Loss> {
  private final int stateCount;
  private final int stateDim;
  private final int alphabetSize;
  private final Alphabet<Type> alphabet;
  private final Random random;
  private final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize, valuesOptimize;
  private final double lambda;
  private final double addToDiag;
  private final int weightValueIterCount;

  public PNFARegressor(
      final int stateCount,
      final int stateDim,
      final Alphabet<Type> alphabet,
      final double lambda,
      final double addToDiag,
      final Random random,
      final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize,
      final Optimize<FuncEnsemble<? extends FuncC1>> valuesOptimize,
      final int weightValueIterCount
  ) {
    this.stateCount = stateCount;
    this.stateDim = stateDim;
    this.alphabetSize = alphabet.size();
    this.lambda = lambda;
    this.addToDiag = addToDiag;
    this.random = random;
    this.weightsOptimize= weightsOptimize;
    this.valuesOptimize = valuesOptimize;
    this.weightValueIterCount = weightValueIterCount;
    this.alphabet = alphabet;
  }

  @Override
  public PNFAModel<Type> fit(final DataSet<Seq<Type>> learn, final Loss loss) {
    Vec params = init(loss.target());
    FuncC1[] funcs = new FuncC1[learn.length()];
    Vec wCacheVec = new ArrayVec(stateCount * stateCount * alphabetSize);
    Mx[] wCache = new Mx[alphabetSize];
    for (int i = 0; i < wCache.length; i++) {
      wCache[i] = new VecBasedMx(stateCount, wCacheVec.sub(stateCount * stateCount * i, stateCount * stateCount));
    }
    for (int i = 0; i < learn.length(); i++) {
      final IntSeq seq = (IntSeq) learn.at(i);
      funcs[i] = new PNFAPointRegression(seq, loss.target().sub(i * stateDim, stateDim), wCache, stateCount, alphabetSize, stateDim, addToDiag);
    }

    weightsOptimize.projector(x -> {
      IntStream.range(0, x.length()).parallel().forEach(idx -> {
        final double val = x.get(idx);
        if (Math.abs(val) > lambda) {
          if (val > lambda) {
            x.adjust(idx, -lambda);
          } else {
            x.adjust(idx, lambda);
          }
        } else {
          x.set(idx, 0);
        }
      });
      VecTools.fill(wCacheVec, -1);
      return x;
    });

    FuncEnsemble<FuncC1> ensemble = new FuncEnsemble<>(funcs, loss.getWeights());
    for (int iter = 0; iter < weightValueIterCount; iter++) {
      VecTools.fill(wCacheVec, -1);
      params = weightsOptimize.optimize(ensemble, params);

      for (int i = 0; i < learn.length(); i++) {
        final IntSeq seq = (IntSeq) learn.at(i);
        funcs[i] = new PNFAPointValueLossFunc(
            seq, loss.target().sub(i* stateDim, stateDim), wCache
        );
      }

      System.out.println("Value before: " + funcs[0].value(params));
      params = valuesOptimize.optimize(ensemble, params);
      System.out.println("Value after: " + funcs[0].value(params));
    }

    return new PNFAModel<>(params, stateCount, stateDim, addToDiag, lambda, alphabet);
  }

  private Vec init(Vec target) {
    final Vec params = new ArrayVec(
        (2 * stateCount) * alphabetSize + stateCount * stateDim
    );
    { // u & v init
      for (int c = 0; c < alphabetSize; c++) {
        for (int i = 0; i < (2 * stateCount) * alphabetSize; i++) {
          params.set(i, random.nextGaussian());
        }
      }
    }

    { // values init
      final Mx values = new VecBasedMx(stateDim, params.sub(params.dim() - stateCount * stateDim, stateCount * stateDim));
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
          values.set(col, i, medians.get(i));
        }
      }
    }

    return params;
  }


  private class PNFAPointValueLossFunc extends FuncC1.Stub {
    private final IntSeq seq;
    private final Vec y;
    private final PNFAPointRegression regression;

    public PNFAPointValueLossFunc(IntSeq seq, Vec y, Mx[] wCache) {
      regression = new PNFAPointRegression(seq, y, wCache, stateCount, alphabetSize, stateDim, addToDiag);
      this.seq = seq;
      this.y = y;
    }

    @Override
    public Vec gradientTo(Vec params, Vec to) {
      Vec distribution = regression.distribution(params);
      final Vec r = VecTools.subtract(
          MxTools.multiply(
              regression.getValues(params), distribution
          ), y
      );
      Mx grad = regression.getValues(to);
      for (int s = 0; s < stateCount; s++) {
        for (int i = 0; i < stateDim; i++) {
          final int idx = s * stateDim + i;
          grad.adjust(i, s, 2 * r.get(i) * distribution.get(s));
        }
      }

      return to;
    }

    @Override
    public double value(Vec params) {
      return regression.value(params);
    }

    @Override
    public int dim() {
      return (2 * stateCount) * alphabetSize + stateCount * stateDim;
    }
  }

  static class PNFAModel<Type> implements Function<Seq<Type>, Vec> {
    private Vec params;
    private int stateCount;
    private int stateDim;
    private double addToDiag;
    private double lambda;
    private Alphabet<Type> alphabet;

    public PNFAModel(Vec params, int stateCount, int stateDim, double addToDiag, double lambda, Alphabet<Type> alpha) {
      this.params = params;
      this.stateCount = stateCount;
      this.stateDim = stateDim;
      this.addToDiag = addToDiag;
      this.lambda = lambda;
      this.alphabet = alpha;
    }

    @Override
    public Vec apply(Seq<Type> seq) {
      PNFAPointRegression regression = new PNFAPointRegression(alphabet.reindex(seq), Vec.EMPTY, null, stateCount, alphabet.size(), stateDim, addToDiag);
      return regression.vecValue(params);
    }

    public Vec getParams() {
      return params;
    }

    public void setParams(Vec params) {
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
}
