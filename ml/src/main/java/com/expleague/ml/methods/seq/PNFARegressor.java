package com.expleague.ml.methods.seq;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.RegularizerFunc;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.methods.seq.param.BettaParametrization;
import com.expleague.ml.methods.seq.param.WeightParametrization;
import com.expleague.ml.optimization.Optimize;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PNFARegressor<Type, Loss extends WeightedL2> implements SeqOptimization<Type, Loss> {
  private static final double W_SIGMA = 0.7;

  private final int startStateCount;
  private final int endStateCount;
  private final int stateDim;
  private final int alphabetSize;
  private final Alphabet<Type> alphabet;
  private final Random random;
  private final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize;
  private final double alpha;
  private final double addToDiag;
  private final double betta;
  private final double expandMxPercent;
  private final BettaParametrization bettaParametrization;
  private final WeightParametrization weightParametrization;

  public PNFARegressor(int startStateCount,
                       int endStateCount,
                       int stateDim,
                       Alphabet<Type> alphabet,
                       double alpha,
                       double betta,
                       double addToDiag,
                       double expandMxPercent,
                       Random random,
                       Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize,
                       BettaParametrization bettaParametrization,
                       WeightParametrization weightParametrization) {
    this.startStateCount = startStateCount;
    this.endStateCount = endStateCount;
    this.betta = betta;
    this.stateDim = stateDim;
    this.alphabetSize = alphabet.size();
    this.alpha = alpha;
    this.addToDiag = addToDiag;
    this.expandMxPercent = expandMxPercent;
    this.random = random;
    this.weightsOptimize= weightsOptimize;
    this.alphabet = alphabet;
    this.bettaParametrization = bettaParametrization;
    this.weightParametrization = weightParametrization;
  }

  @Override
  public PNFAModel<Type> fit(final DataSet<Seq<Type>> learn, final Loss loss) {
    Vec params = init(loss.target(), startStateCount);
    PNFAItemVecRegression[] funcs = new PNFAItemVecRegression[learn.length()];

    //Vec wCacheVec = new ArrayVec(stateCount * stateCount * alphabetSize);
    //VecTools.fill(wCacheVec, -1);
//    Mx[] wCache = new Mx[alphabetSize];

//    for (int i = 0; i < wCache.length; i++) {
//      wCache[i] = new VecBasedMx(stateCount, wCacheVec.sub(stateCount * stateCount * i, stateCount * stateCount));
//    }


    for (int stateCount = startStateCount; stateCount <= endStateCount; stateCount++) {
      for (int i = 0; i < learn.length(); i++) {
        final IntSeq seq = (IntSeq) learn.at(i);
        funcs[i] = new PNFAItemVecRegression(
            seq,
            loss.target().sub(i * stateDim, stateDim),
            stateCount,
            alphabetSize,
            stateDim,
            bettaParametrization,
            weightParametrization
        );
      }

      final RegularizerFunc regularizer = new MyRegularizer(funcs, null, stateCount, alpha, betta);
      final FuncEnsemble<FuncC1> func = new FuncEnsemble<>(funcs, loss.getWeights());

      params = weightsOptimize.optimize(func, regularizer, params);

      double totalEntropy = 0;
      for (int i =0 ; i < learn.length(); i++) {
        totalEntropy += VecTools.entropy(funcs[i].distribution(params));
      }
      System.out.println("Entropy: " + (totalEntropy / learn.length()));
      if (stateCount == endStateCount) {
        break;
      }


      List<Integer> wIdx = IntStream.range(0, alphabetSize).boxed().collect(Collectors.toList());
      final Vec paramsFinal = params;
      final int stateCountFinal = stateCount;
      wIdx.sort(Comparator.comparingDouble(i -> {
            Mx w = weightParametrization.getMx(paramsFinal, i, stateCountFinal);
            double entropy = Double.MAX_VALUE;
            for (int j = 0; j < w.columns(); j++) {
              entropy = Math.min(entropy, VecTools.entropy(w.col(j)));
            }
            return entropy;
          }
      ));

      Set<Integer> badMxIdx = wIdx.stream().limit((int) (expandMxPercent * wIdx.size())).collect(Collectors.toSet());

      Vec newParams = new ArrayVec(2 * (stateCount + 1) * alphabetSize + (stateCount + 1) * stateDim);

      int paramsPos = 0, newParamsPos = 0;

      //todo generify for different parametrizations
      for (int a = 0; a < alphabetSize; a++, paramsPos += 2 * stateCount, newParamsPos += 2 * (stateCount + 1)) {
        VecTools.assign(newParams.sub(newParamsPos, stateCount), params.sub(paramsPos, stateCount));
        VecTools.assign(newParams.sub(newParamsPos + stateCount + 1, stateCount), params.sub(paramsPos + stateCount, stateCount));

        newParams.set(newParamsPos + 2 * stateCount + 1, W_SIGMA * random.nextGaussian());

        double val;
        if (badMxIdx.contains(a)) {
          val = W_SIGMA * random.nextGaussian() * 100;
        }
        else {
          val = W_SIGMA * random.nextGaussian();
        }
        newParams.set(newParamsPos + stateCount, val);
      }

      Vec expected = newParams.sub(newParams.length() - stateDim, stateDim);
      for (int i = 0; i < learn.length(); i++) {
        final IntSeq seq = (IntSeq) learn.at(i);
        PNFAItemVecRegression f = new PNFAItemVecRegression(
            seq,
            loss.target().sub(i * stateDim, stateDim),
            stateCount + 1,
            alphabetSize,
            stateDim,
            bettaParametrization,
            weightParametrization
        );
        Vec distr = f.distribution(newParams);
        for (int j = 0; j < stateDim; j++) {
          expected.adjust(j, distr.get(j) * loss.target().get(i * stateDim + j) / learn.length());
        }
      }

      params = newParams;
    }

      //    System.out.println("Value after: " + func.value(params) / func.size());

    return new PNFAModel<>(params, endStateCount, stateDim, addToDiag, alpha, alphabet, bettaParametrization, weightParametrization);
  }


  private Vec init(Vec target, int stateCount) {
    int paramCount = bettaParametrization.paramCount(stateCount);
    final Vec params = new ArrayVec(
        paramCount * alphabetSize + stateCount * stateDim
    );
    { // u & v init
      for (int i = 0; i < paramCount * alphabetSize; i++) {
        params.set(i, W_SIGMA * Math.abs(random.nextGaussian()));
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

  private class MyRegularizer extends RegularizerFunc.Stub {
    private final PNFAItemVecRegression[] funcs;
    private final int stateCount;
    private final double alpha;
    private final double betta;
    private final Vec wCacheVec;
    Vec prev;

    public MyRegularizer(PNFAItemVecRegression[] funcs, Vec wCacheVec, int stateCount, double alpha, double betta) {
      this.funcs = funcs;
      this.wCacheVec = wCacheVec;
      this.stateCount = stateCount;
      this.alpha = alpha;
      this.betta = betta;
      prev = null;
      prev = new ArrayVec(funcs[0].dim());
    }

    @Override
    public double value(Vec x) {
      int paramCount = bettaParametrization.paramCount(stateCount);
      return alpha * VecTools.l1(x.sub(0, paramCount * alphabetSize))
          + betta * VecTools.l2(x.sub(paramCount * alphabetSize, stateCount * stateDim));
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
}
