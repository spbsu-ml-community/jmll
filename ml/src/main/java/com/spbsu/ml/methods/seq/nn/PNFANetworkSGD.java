package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.SeqOptimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PNFANetworkSGD<T, Loss extends L2> implements SeqOptimization<T, Loss> {
  private final int stateCount;
  private final Alphabet<T> alphabet;
  private final Random random;
  private final double step;
  private final int iterationsCount;

  public PNFANetworkSGD(final Alphabet<T> alphabet,
                        final int stateCount,
                        final Random random,
                        final double step,
                        final int iterationsCount) {
    this.stateCount = stateCount;
    this.alphabet = alphabet;
    this.random = random;
    this.step = step;
    this.iterationsCount = iterationsCount;
  }

  @Override
  public Computable<Seq<T>, Vec> fit(DataSet<Seq<T>> learn, Loss loss) {
    final PNFAParams<T> params = new PNFAParams<>(random, stateCount, alphabet);

    int maxLen = 0;
    for (int i = 0; i < learn.length(); i++) {
      maxLen = Math.max(maxLen, learn.at(i).length());
    }

    final int[][] seqAlphabet = new int[learn.length()][];

    for (int i = 0; i < learn.length(); i++) {
      final Seq<T> seq = learn.at(i);
      List<Integer> charsList = new ArrayList<>(seq.length());
      for (int j = 0; j < seq.length(); j++) {
        charsList.add(alphabet.index(seq.at(j)));
      }
      seqAlphabet[i] = charsList.stream().sorted().distinct().mapToInt(x -> x).toArray();
    }

    double curCost = getCost(learn, loss, params);
    long start = System.nanoTime();

    for (int iter = 0; iter < iterationsCount; iter++) {
      final int seqId = random.nextInt(learn.length());
      final Seq<T> seq = learn.at(seqId);

      final PNFAParams.PNFAParamsGrad grad = params.calcSeqGrad(seq, seqAlphabet[seqId], loss.target().at(seqId));
      params.updateParams(grad.getWGrad(), grad.getValuesGrad(), -step, seqAlphabet[seqId]);

      if (iter % (learn.length() * 10) == 0) {
        long cur = System.nanoTime();
        final double newCost = getCost(learn, loss, params);
        //System.out.printf("Iteration %d, cost=%.6f\n", iter, newCost);
      //  System.out.printf("Iterations elapsed  %d, cost=%.6f, 100 iterations passed in %.2f minutes\n", iter, curCost, (cur - start) / 60e9);
        start = cur;
        if (newCost > curCost && iter != 0) {
          System.out.printf("Iterations elapsed %d, cost=%.6f\n", iter, curCost);
          break;
        }
        System.out.flush();
        curCost= newCost;
      }
    }

    return (seq) -> new SingleValueVec(params.getSeqValue(seq));
  }

  private double getCost(final DataSet<Seq<T>> learn, final Loss loss, final PNFAParams<T> params) {
    double cost = 0;
    for (int i = 0; i < learn.length(); i++) {
      cost += MathTools.sqr(params.getSeqValue(learn.at(i)) - loss.target().at(i));
    }
    return cost / learn.length();
  }
}
