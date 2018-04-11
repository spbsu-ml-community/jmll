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
import com.expleague.ml.func.ReguralizerFunc;
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
  private final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize;
  private final double alpha;
  private final double addToDiag;
  private final double betta;

  public PNFARegressor(int stateCount, int stateDim, Alphabet<Type> alphabet, double alpha, double betta, double addToDiag, Random random, Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize) {
    this.stateCount = stateCount;
    this.betta = betta;
    this.stateDim = stateDim;
    this.alphabetSize = alphabet.size();
    this.alpha = alpha;
    this.addToDiag = addToDiag;
    this.random = random;
    this.weightsOptimize= weightsOptimize;
    this.alphabet = alphabet;
  }

  @Override
  public PNFAModel<Type> fit(final DataSet<Seq<Type>> learn, final Loss loss) {
    Vec params = init(loss.target());
    PNFAItemVecRegression[] funcs = new PNFAItemVecRegression[learn.length()];
    Vec wCacheVec = new ArrayVec(stateCount * stateCount * alphabetSize);
    VecTools.fill(wCacheVec, -1);
    Mx[] wCache = new Mx[alphabetSize];
    for (int i = 0; i < wCache.length; i++) {
      wCache[i] = new VecBasedMx(stateCount, wCacheVec.sub(stateCount * stateCount * i, stateCount * stateCount));
    }
    for (int i = 0; i < learn.length(); i++) {
      final IntSeq seq = (IntSeq) learn.at(i);
      funcs[i] = new PNFAItemVecRegression(seq, loss.target().sub(i * stateDim, stateDim), wCache, stateCount, alphabetSize, stateDim, addToDiag);
    }

    final ReguralizerFunc reguralizerFunc = new MyReguralizer(funcs, wCacheVec, alpha, betta);
    final FuncEnsemble<FuncC1> func = new FuncEnsemble<>(funcs, loss.getWeights());
    params = weightsOptimize.optimize(func, reguralizerFunc, params);
//    System.out.println("Value after: " + func.value(params) / func.size());

    return new PNFAModel<>(params, stateCount, stateDim, addToDiag, alpha, alphabet);
  }

  private Vec init(Vec target) {
    final Vec params = new ArrayVec(
        (2 * stateCount) * alphabetSize + stateCount * stateDim
    );
    { // u & v init
      for (int c = 0; c < alphabetSize; c++) {
        for (int i = 0; i < (2 * stateCount) * alphabetSize; i++) {
          params.set(i, 0.1 * Math.abs(random.nextGaussian()));
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
          values.set(i, col, medians.get(i));
        }
      }
    }

    return params;
  }

  private class MyReguralizer extends ReguralizerFunc.Stub {
    private final PNFAItemVecRegression[] funcs;
    private final double alpha;
    private final double betta;
    private final Vec wCacheVec;
    Vec prev;

    public MyReguralizer(PNFAItemVecRegression[] funcs, Vec wCacheVec, double alpha, double betta) {
      this.funcs = funcs;
      this.wCacheVec = wCacheVec;
      this.alpha = alpha;
      this.betta = betta;
      prev = null;
      prev = new ArrayVec(funcs[0].dim());
    }

    @Override
    public double value(Vec x) {
      return alpha * VecTools.l1(x.sub(0, 2 * stateCount * alphabetSize)) + betta * VecTools.l2(x.sub(2 * stateCount * alphabetSize, stateCount * stateDim));
    }

    @Override
    public int dim() {
      return (2 * stateCount) * alphabetSize + stateCount * stateDim;
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
      VecTools.fill(wCacheVec, -1);
      return x;
    }
  }
}
