package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.regexp.Alphabet;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.SeqOptimization;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static com.spbsu.commons.math.vectors.VecTools.*;

public class PNFANetworkGD<T, Loss extends L2> implements SeqOptimization<T, Loss> {
  private final int stateCount;
  private final Alphabet<T> alphabet;
  private final FastRandom random;
  private final double step;
  private final int iterationsCount;
  private final int threadCount;

  public PNFANetworkGD(final Alphabet<T> alphabet,
                       final int stateCount,
                       final FastRandom random,
                       final double step,
                       final int iterationsCount,
                       final int threadCount) {
    this.stateCount = stateCount;
    this.alphabet = alphabet;
    this.random = random;
    this.step = step;
    this.iterationsCount = iterationsCount;
    this.threadCount = threadCount;
  }

  @Override
  public Computable<Seq<T>, Vec> fit(DataSet<Seq<T>> learn, Loss loss) {
    final PNFAParams<T> params = new PNFAParams<>(random, stateCount, alphabet);
    final Vec[] distribs = new Vec[learn.length()];
    for (int i = 0; i < distribs.length; i++) {
      distribs[i] = new ArrayVec(stateCount);
      fill(distribs[i], 1./distribs[i].dim());
    }

    long start = System.nanoTime();
    final Mx[] bGradients = new Mx[alphabet.size()];
    final Vec vGradients = new ArrayVec(stateCount);

    for (int i = 0; i < bGradients.length; i++) {
      bGradients[i] = new VecBasedMx(stateCount, stateCount - 1);
    }
    params.updateParams(bGradients, vGradients, learn, loss.target(), distribs, 0);
    double curCost = getCost(learn, loss, params);

    for (int iter = 0; iter < iterationsCount; iter++) {
      fill(vGradients, 0);
      for (final Mx bGradient : bGradients) {
        fill(bGradient, 0);
      }

      final TIntArrayList path = new TIntArrayList(1000);
      final TDoubleArrayList transitions = new TDoubleArrayList(1000);
      final int samplesCount = learn.length();
      for (int i = 0; i < samplesCount; i++) {
        final int sampleIdx = random.nextInt(learn.length());
        final Seq<T> sample = learn.at(sampleIdx);
        int currentState = 0;
        path.clear();
        transitions.clear();
        for (int t = 0; t < sample.length(); t++) {
          final Vec weights = params.getW()[alphabet.index(sample.at(t))].row(currentState);
          currentState = random.nextSimple(weights);
          path.add(currentState);
          transitions.add(weights.get(currentState));
        }
        final double finalStateValue = params.getValues().get(currentState);
        final double pLastState = distribs[sampleIdx].get(currentState);
        double residual = 2 * (loss.target().get(sampleIdx) - multiply(distribs[sampleIdx], params.getValues()));
        for (int k = 0; k < vGradients.length(); k++) {
          vGradients.adjust(k, residual*distribs[sampleIdx].get(k));
        }
        residual *= finalStateValue;
        for (int t = sample.length() - 1; t >= 0; t--) {
          currentState = path.get(t);
          final int prevState = t > 0 ? path.get(t - 1) : 0;
          final double prevProb = t > 0 ? transitions.get(t - 1) : 1;
          final Vec weights = params.getW()[alphabet.index(sample.at(t))].row(prevState);
          final Vec vec = bGradients[alphabet.index(sample.at(t))].row(prevState);
          final double rresidual = residual*prevProb/weights.get(currentState);

          for (int j = 0; j < stateCount - 1; j++) {
            final double w = weights.get(j);
            if (j == currentState) {
              vec.adjust(j, rresidual * w * (1 - w));
            }
            else {
              vec.adjust(j, -rresidual * w);
            }
          }
        }
      }
      if (iter % 500 < 10 || iter % 500 > 460) {
        for (final Mx bGradient : bGradients) {
          fill(bGradient, 0);
        }
      }
      else fill(vGradients, 0);

      // step
      params.updateParams(bGradients, vGradients, learn, loss.target(), distribs, 0.7/samplesCount);
      double totalGrad = 0;
      for (int i = 0; i < bGradients.length; i++) {
        totalGrad += VecTools.norm(bGradients[i]);
      }
      if (iter % 500 == 0) {
        long cur = System.nanoTime();
        double newCost = 0;
        for (int i = 0; i < learn.length(); i++) {
          final double sqr = MathTools.sqr(multiply(distribs[i], params.getValues()) - loss.target().at(i));
          newCost += sqr;
        }
        newCost /= learn.length();

        //System.out.printf("Iteration %d, cost=%.6f\n", iter, newCost);
        System.out.printf("Iterations elapsed  %d, cost=%.6f, 100 iterations passed in %.2f minutes\n", iter, curCost, (cur - start) / 60e9);
        System.out.flush();
        start = cur;
        if (newCost > curCost) {
          System.out.printf("Iterations elapsed %d, cost=%.6f\n", iter, newCost);
          System.out.println("Values: " + params.getValues());
          Vec weights = new ArrayVec(stateCount);
          for (int i = 0; i < loss.dim(); i++) {
            append(weights, distribs[i]);
          }
          System.out.println("Weights: " + weights);
          break;
        }
        curCost= newCost;
      }
    }

    return (seq) -> new SingleValueVec(params.getSeqValue(seq));
  }

  private double getCost(final DataSet<Seq<T>> learn, final Loss loss, final PNFAParams<T> params) {
    double cost = 0;
    for (int i = 0; i < learn.length(); i++) {
      final double sqr = MathTools.sqr(params.getSeqValue(learn.at(i)) - loss.target().at(i));
      cost += sqr;
    }
    return cost / learn.length();
  }
}
