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
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.SeqOptimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

public class PNFANetworkGD<T, Loss extends L2> implements SeqOptimization<T, Loss> {
  private final int stateCount;
  private final Alphabet<T> alphabet;
  private final Random random;
  private final double step;
  private final int iterationsCount;
  private final int threadCount;

  public PNFANetworkGD(final Alphabet<T> alphabet,
                       final int stateCount,
                       final Random random,
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
    final PNFAParams<T>[] taskParams = new PNFAParams[threadCount];

    int maxLen = 0;
    for (int i = 0; i < learn.length(); i++) {
      maxLen = Math.max(maxLen, learn.at(i).length());
    }

    final int[] path = new int[maxLen + 1];
    for (int i = 0; i < threadCount; i++) {
      taskParams[i] = new PNFAParams<>(random, stateCount, alphabet);
    }

    final ExecutorService executorService = Executors.newFixedThreadPool(threadCount);

    final Mx[] totalWGrad = new Mx[alphabet.size()];
    final Vec totalValuesGrad = new ArrayVec(stateCount);
    for (int i = 0; i < alphabet.size(); i++) {
      totalWGrad[i] = new VecBasedMx(stateCount, stateCount);
    }

    double curCost = getCost(learn, loss, params);
    long start = System.nanoTime();

    for (int iter = 0; iter < iterationsCount; iter++) {
      for (int i = 0; i < alphabet.size(); i++) {
        VecTools.fill(totalWGrad[i], 0);
      }
      VecTools.fill(totalValuesGrad, 0);

      final List<Future<Vec>> pathFutures = new ArrayList<>(threadCount);
      List<Seq<T>> curSeqs = new ArrayList<>(learn.length());
      for (int i = 0; i < learn.length(); i++) {
        curSeqs.add(learn.at(i));
      }

      path[0] = 0;
      for (int posInSeq = 0; posInSeq < maxLen; posInSeq++) {
        pathFutures.clear();
        final int posInSeq1 = posInSeq;
        curSeqs = curSeqs.stream().filter(seq -> posInSeq1 < seq.length()).collect(Collectors.toList());
        for (int taskId = 0; taskId < threadCount; taskId++) {
          final int taskId1 = taskId;
          final int samplesPerThread = (curSeqs.size() + threadCount - 1) / threadCount;
          final int startSub = samplesPerThread * taskId1;
          final int endSub = Math.min(curSeqs.size()  , (taskId1 + 1) * samplesPerThread);
          final List<Seq<T>> curSeqs1 = curSeqs;

          taskParams[taskId1].setParams(params);
          pathFutures.add(executorService.submit(() -> taskParams[taskId1].getNextStateDistribution(
                  path[posInSeq1], posInSeq1, curSeqs1, startSub, endSub
          )));
        }

        final Vec distribution = new ArrayVec(stateCount);
        pathFutures.stream().forEach(future -> {
          try {
            VecTools.append(distribution, future.get());
          } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
          }
        });

        VecTools.normalizeL1(distribution);
        double rnd = random.nextDouble();
        path[posInSeq + 1] = stateCount - 1;
        for (int to = 0; to < stateCount; to++) {
          if (rnd <= distribution.at(to)) {
            path[posInSeq + 1] = to;
            break;
          }
          rnd -= distribution.at(to);
        }
      }

      final List<Future<PNFAParams.PNFAParamsGrad>> gradFutures = new ArrayList<>(threadCount);

      for (int taskId = 0; taskId < threadCount; taskId++) {
        taskParams[taskId].setParams(params);
        final int samplesPerThread = (learn.length() + threadCount - 1) / threadCount;
        final int taskId1 = taskId;
        final int startSub = samplesPerThread * taskId;
        final int endSub = Math.min(learn.length(), (taskId + 1) * samplesPerThread);

        gradFutures.add(executorService.submit(() -> taskParams[taskId1].calcPathGrad(
                learn.sub(startSub, endSub),
                loss.target().sub(startSub, endSub - startSub),
                path)
        ));
      }

      gradFutures.stream().forEach(future -> {
        try {
          final PNFAParams.PNFAParamsGrad grad = future.get();
          for (int i = 0; i < alphabet.size(); i++) {
            VecTools.append(totalWGrad[i], grad.getWGrad()[i]);
          }
          VecTools.append(totalValuesGrad, grad.getValuesGrad());
        } catch (InterruptedException | ExecutionException e) {
          e.printStackTrace();
        }
      });

      params.updateParams(totalWGrad, totalValuesGrad, -step / learn.length());

      if (iter % 500 == 0) {
        long cur = System.nanoTime();
        final double newCost = getCost(learn, loss, params);
        //System.out.printf("Iteration %d, cost=%.6f\n", iter, newCost);
        System.out.printf("Iterations elapsed  %d, cost=%.6f, 500 iterations passed in %.2f minutes\n", iter, curCost, (cur - start) / 60e9);
        System.out.flush();
        start = cur;
        if (newCost > curCost) {
          System.out.printf("Iterations elapsed %d, cost=%.6f\n", iter, newCost);
          System.out.println("Values: " + params.getValues());
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
      cost += MathTools.sqr(params.getSeqValue(learn.at(i)) - loss.target().at(i));
    }
    return cost / learn.length();
  }
}
