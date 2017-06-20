package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.SeqOptimization;
import com.spbsu.ml.methods.seq.automaton.DFA;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PNFANetworkSAGA<T, Loss extends L2> implements SeqOptimization<T, Loss> {
  private final int stateCount;
  private final Alphabet<T> alphabet;
  private final Random random;
  private final double step;
  private final int iterationsCount;

  public PNFANetworkSAGA(final Alphabet<T> alphabet,
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

    final Vec[] lastValuesGrad = new Vec[learn.length()];
    final Mx[][] lastBetaGrad = new Mx[learn.length()][];
    final int[] lastAlphaGradChanged = new int[alphabet.size()];
    final int[][] seqAlphabet = new int[learn.length()][];

    final Mx[] totalBetaGrad = new Mx[alphabet.size()];
    for (int i = 0; i < alphabet.size(); i++) {
      totalBetaGrad[i] = new VecBasedMx(stateCount, stateCount - 1);
    }
    final Vec totalValuesGrad = new ArrayVec(stateCount);

    final Vec valuesGrad = new ArrayVec(stateCount);
    final Mx[] betaGrad = new Mx[alphabet.size()];
    for (int i = 0; i < alphabet.size(); i++) {
      betaGrad[i] = new VecBasedMx(stateCount, stateCount - 1);
    }

    for (int i = 0; i < learn.length(); i++) {
      final Seq<T> seq = learn.at(i);
      List<Integer> charsList = new ArrayList<>(seq.length());
      for (int j = 0; j < seq.length(); j++) {
        charsList.add(alphabet.index(seq.at(j)));
      }
      seqAlphabet[i] = charsList.stream().sorted().distinct().mapToInt(x -> x).toArray();
      lastBetaGrad[i] = new Mx[seqAlphabet[i].length];
      final PNFAParams.PNFAParamsGrad grad = params.calcSeqGrad(seq, seqAlphabet[i], loss.target().at(i));
      lastBetaGrad[i] = grad.getBetaGrad();
      lastValuesGrad[i] = grad.getValuesGrad();

      VecTools.append(totalValuesGrad, grad.getValuesGrad());
      for (int j = 0; j < seqAlphabet[i].length; j++) {
        final int c = seqAlphabet[i][j];
        VecTools.append(totalBetaGrad[c], grad.getBetaGrad()[j]);
      }
    }

    for (int c = 0; c < alphabet.size(); c++) {
      VecTools.incscale(betaGrad[c], totalBetaGrad[c], 1.0 / learn.length());
    }

    VecTools.incscale(valuesGrad, totalValuesGrad, 1.0 / learn.length());

    double curCost = getCost(learn, loss, params);
    long start = System.nanoTime();

    final Mx[] seqBetaGrad = new Mx[maxLen];
    Vec seqValuesGrad = new ArrayVec(stateCount);

    for (int iter = 1; iter < iterationsCount; iter++) {
      final int seqId = random.nextInt(learn.length());
      final Seq<T> seq = learn.at(seqId);

      final PNFAParams.PNFAParamsGrad grad = params.calcSeqGrad(seq, seqAlphabet[seqId], loss.target().at(seqId));

      for (int i = 0; i < seqAlphabet[seqId].length; i++) {
        final int a = seqAlphabet[seqId][i];
        seqBetaGrad[i] = betaGrad[a];
        VecTools.fill(seqBetaGrad[i], 0);
        // character a was not counted during last (iter - lastAlphaGradChanged[a] - 1) iterations as part of
        // the average sum of derivatives
        VecTools.incscale(betaGrad[a], totalBetaGrad[a], (double) (iter - lastAlphaGradChanged[a]) / learn.length());
        lastAlphaGradChanged[a] = iter;

        VecTools.append(betaGrad[a], grad.getBetaGrad()[i]);
        VecTools.incscale(betaGrad[a], lastBetaGrad[seqId][i], -1);

        VecTools.incscale(totalBetaGrad[a], lastBetaGrad[seqId][i], -1);
        VecTools.append(totalBetaGrad[a], grad.getBetaGrad()[i]);
        lastBetaGrad[seqId][i] = grad.getBetaGrad()[i];
      }

      VecTools.fill(valuesGrad, 0);
      VecTools.append(valuesGrad, grad.getValuesGrad());
      VecTools.incscale(valuesGrad, lastValuesGrad[seqId], -1);
      VecTools.incscale(valuesGrad, totalValuesGrad, 1.0 / learn.length());

      VecTools.incscale(totalValuesGrad, lastValuesGrad[seqId], -1);
      VecTools.append(totalValuesGrad, grad.getValuesGrad());
      lastValuesGrad[seqId] = grad.getValuesGrad();

      // do not apply update to all the characters
      params.updateParams(seqBetaGrad, valuesGrad, -step, seqAlphabet[seqId]);

      if (iter % (5 * learn.length()) == 0) {
        long cur = System.nanoTime();

        for (int seqId1 = 0; seqId1 < learn.length(); seqId1++) {
          for (int i = 0; i < seqAlphabet[seqId1].length; i++) {
            final int a = seqAlphabet[seqId1][i];
            seqBetaGrad[i] = betaGrad[a];
            VecTools.fill(seqBetaGrad[i], 0);
            // character a was not counted during last (iter - lastAlphaGradChanged[a] - 1) iterations as part of
            // the average sum of derivatives
            VecTools.incscale(seqBetaGrad[i], totalBetaGrad[a], (double) (iter - lastAlphaGradChanged[a]) / learn.length());
            lastAlphaGradChanged[a] = iter;
          }
          VecTools.fill(valuesGrad, 0);
          params.updateParams(seqBetaGrad, valuesGrad, -step, seqAlphabet[seqId1]);
        }
        final double newCost = getCost(learn, loss, params);
            System.out.printf("Iteration %d, cost=%.6f\n", iter, newCost);
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
    DFA<T> result = new DFA<>(alphabet);
    for (int i = 1; i < stateCount; i++) result.createNewState();

    for (int i = 0; i < alphabet.size(); i++) {
      for (int j = 0; j < stateCount; j++) {
        int id = ArrayTools.max(params.getW()[i].row(j).toArray());
        result.addTransition(j, id, alphabet.getT(alphabet.get(i)));
      }
    }
//    return (seq) -> new SingleValueVec(params.getValues().get(result.run(seq))); //new SingleValueVec(params.getSeqValue(seq));

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
