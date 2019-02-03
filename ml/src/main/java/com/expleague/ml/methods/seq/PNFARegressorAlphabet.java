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

public class PNFARegressorAlphabet<Type, Loss extends WeightedL2> implements SeqOptimization<Type, Loss> {
  private static final double W_SIGMA = 0.7;

  private final int startAlphabetSize;
  private final int endAlphabetSize;
  private final int stateCount;
  private final int stateDim;
  private final Alphabet<Type> finalAlphabet;
  private final Random random;
  private final Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize;
  private final double alpha;
  private final double addToDiag;
  private final double betta;
  private final BettaParametrization bettaParametrization;
  private final WeightParametrization weightParametrization;

  public PNFARegressorAlphabet(int startAlphabetSize,
                               int endAlphabetSize,
                               int stateCount,
                               int stateDim,
                               Alphabet<Type> finalAlphabet,
                               double alpha,
                               double betta,
                               double addToDiag,
                               Random random,
                               Optimize<FuncEnsemble<? extends FuncC1>> weightsOptimize,
                               BettaParametrization bettaParametrization,
                               WeightParametrization weightParametrization) {
    this.startAlphabetSize = startAlphabetSize;
    this.endAlphabetSize = endAlphabetSize;
    this.betta = betta;
    this.stateDim = stateDim;
    this.stateCount = stateCount;
    this.alpha = alpha;
    this.addToDiag = addToDiag;
    this.random = random;
    this.weightsOptimize= weightsOptimize;
    this.finalAlphabet = finalAlphabet;
    this.bettaParametrization = bettaParametrization;
    this.weightParametrization = weightParametrization;
  }

  @Override
  public PNFAModel<Type> fit(final DataSet<Seq<Type>> learn, final Loss loss) {
    Vec params = init(loss.target(), stateCount);
    PNFAItemVecRegression[] funcs = new PNFAItemVecRegression[learn.length()];

    for (int alphabetSize = startAlphabetSize; alphabetSize <= endAlphabetSize; alphabetSize++) {
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

      //final RegularizerFunc regularizer = new MyRegularizer(funcs, null, stateCount, alpha, betta, bettaParametrization, alphabetSize, stateDim);
      final FuncEnsemble<FuncC1> func = new FuncEnsemble<>(funcs, loss.getWeights());

      params = weightsOptimize.optimize(func, params);

      double totalEntropy = 0;
      for (int i =0 ; i < learn.length(); i++) {
        totalEntropy += VecTools.entropy(funcs[i].distribution(params));
      }
      System.out.println("Entropy: " + (totalEntropy / learn.length()));
      if (alphabetSize == endAlphabetSize) {
        break;
      }

      for (int i = 0; i < alphabetSize; i++) {

        printMx(weightParametrization.getMx(params, i, stateCount));
        System.out.println("=========");
      }


      int paramCount = bettaParametrization.paramCount(stateCount);
      Vec newParams = new ArrayVec(paramCount * (alphabetSize + 1) + stateCount * stateDim);
      for (int i = 0; i < paramCount * alphabetSize; i++) {
        newParams.set(i, params.get(i));
      }
      for (int i = paramCount * alphabetSize; i < paramCount * (alphabetSize + 1); i++) {
        params.set(i, W_SIGMA * Math.abs(random.nextGaussian()));
      }
      for (int i = paramCount * (alphabetSize + 1); i < newParams.dim(); i++) {
        newParams.set(i, params.get(i - paramCount));
      }


      params = newParams;
    }

    return new PNFAModel<>(params, stateCount, stateDim, addToDiag, alpha, finalAlphabet, bettaParametrization, weightParametrization);
  }


  private Vec init(Vec target, int stateCount) {
    int paramCount = bettaParametrization.paramCount(stateCount);
    final Vec params = new ArrayVec(
        paramCount * startAlphabetSize + stateCount * stateDim
    );
    { // u & v init
      for (int i = 0; i < paramCount * startAlphabetSize; i++) {
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

  private void printMx(Mx mx) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++) {
        System.out.printf("%.3f ", mx.get(i, j));
      }
      System.out.println();
    }
  }
}
