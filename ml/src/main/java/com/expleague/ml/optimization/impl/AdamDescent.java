package com.expleague.ml.optimization.impl;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.optimization.Optimize;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class AdamDescent implements Optimize<FuncEnsemble<? extends FuncC1>> {

  private final double step;
  private final double beta1;
  private final double beta2;
  private final double eps;
  private final Random random;
  private final int epochCount;
  private final int batchSize;

  public AdamDescent(Random random, int epochCount, int batchSize) {
    this(random, epochCount, batchSize, 0.001, 0.9, 0.999, 1e-8);
  }

  public AdamDescent(Random random, int epochCount, int batchSize, double step) {
    this(random, epochCount, batchSize, step, 0.9, 0.999, 1e-8);
  }

  public AdamDescent(Random random, int epochCount, int batchSize, double step, double beta1, double beta2, double eps) {
    this.random = random;
    this.epochCount = epochCount;
    this.batchSize = batchSize;
    this.step = step * Math.sqrt(batchSize);
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
  }

  @Override
  public Vec optimize(FuncEnsemble<? extends FuncC1> sumFuncs) {
    final Vec x = new ArrayVec(sumFuncs.dim());
    for (int i = 0; i < sumFuncs.dim(); i++) {
      x.set(i, random.nextGaussian());
    }
    return optimize(sumFuncs, x);
  }

  @Override
  public Vec optimize(FuncEnsemble<? extends FuncC1> sumFuncs, Vec x0) {
    final long startTime = System.nanoTime();

    final Vec x = VecTools.copy(x0);
    final Vec v = new ArrayVec(x.dim());
    final Vec c = new ArrayVec(x.dim());
    double error = sumFuncs.value(x) / sumFuncs.size();

    final List<Integer> permutation = new ArrayList<>(sumFuncs.size());
    for (int i = 0; i < sumFuncs.size(); i++) {
      permutation.add(i);
    }

    long timeToGrad = 0;
    long timeToSum = 0;
    for (int epoch = 0; epoch < epochCount; epoch++) {
      Collections.shuffle(permutation, random);
      for (int i = 0; i + batchSize < sumFuncs.size(); i += batchSize) {
        IntStream stream;
        if (batchSize > 1) {
          stream = IntStream.range(i, i + batchSize).parallel();
        } else {
          stream = IntStream.range(i, i + batchSize);
        }
        long start = System.nanoTime();
        final double[] grad = stream
            .mapToObj(j -> sumFuncs.models[permutation.get(j)].gradient(x))
            .reduce(VecTools::append).get().toArray();
        //todo is converting sparse vector to array a good idea?
        timeToGrad += System.nanoTime() - start;
        start = System.nanoTime();

        final int threadCount = Runtime.getRuntime().availableProcessors();
        final int blockSize = (x.dim() + threadCount - 1) / threadCount;
        IntStream.range(0, threadCount).parallel().forEach(blockNum -> {
          for (int index = blockNum * blockSize; index < Math.min(x.dim(), (blockNum + 1) * blockSize); index++) {
            final double gradAt = grad[index] / batchSize;
            v.set(index, v.get(index) * beta2 + gradAt * (1 - beta2));
            c.set(index, c.get(index) * beta1 + MathTools.sqr(gradAt * (1 - beta1)));
            x.adjust(index, -step * v.get(index) / (Math.sqrt(c.get(index) + eps)));
          }
        });
        timeToSum += System.nanoTime() - start;
      }
      if ((epoch + 1) % 5 == 0) {
        final double curError = sumFuncs.value(x) / sumFuncs.size();
        System.out.printf("ADAM descent epoch %d: new=%.6f old=%.6f\n", epoch, curError, error);
        if (curError > error) {
          System.out.printf("ADAM descent finished after %d epochs\n", epoch);
          break;
        }
        error = curError;
      } else if (epoch == epochCount - 1) {
        final double curError = sumFuncs.value(x) / sumFuncs.size();
        System.out.printf("ADAM descent epoch %d: new=%.6f old=%.6f\n", epoch, curError, error);
      }
    }
    System.out.printf("Time to grad: %.3f, time to sum: %.3f\n", timeToGrad / 1e9, timeToSum / 1e9);
    System.out.printf("Adam Descent finished in %.2f seconds\n", (System.nanoTime() - startTime) / 1e9);
    return x;
  }
}
