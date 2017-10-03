package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.optimization.Optimize;

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
    this(random, epochCount, batchSize, 0.002, 0.9, 0.999, 1e-8);
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
    Vec grad = new ArrayVec(x.dim());
    double error = sumFuncs.value(x) / sumFuncs.size();

    final List<Integer> permutation = new ArrayList<>(sumFuncs.size());
    for (int i = 0; i < sumFuncs.size(); i++) {
      permutation.add(i);
    }

    for (int epoch = 0; epoch < epochCount; epoch++) {
      Collections.shuffle(permutation, random);
      for (int i = 0; i + batchSize < sumFuncs.size(); i += batchSize) {
        VecTools.fill(grad, 0);
        IntStream stream;
        if (batchSize > 1) {
          stream = IntStream.range(i, i + batchSize).parallel();
        } else {
          stream = IntStream.range(i, i + batchSize);
        }
        grad = stream
            .mapToObj(j -> VecTools.scale(sumFuncs.models[permutation.get(j)].gradient(x), 1.0 / batchSize))
            .reduce((vec1, vec2) -> VecTools.append(vec1, vec2)).get();

//        for (int j = i; j < i + batchSize; j++) {
//          VecTools.append(grad, sumFuncs.models[permutation.get(j)].gradient(x));
//        }
//        VecTools.scale(grad, 1.0 / batchSize);
        VecTools.scale(v, beta2);
        VecTools.incscale(v, grad, 1 - beta2);

        VecTools.scale(c, beta1);
        VecTools.scale(grad, grad);
        VecTools.incscale(c, grad, 1 - beta1);

        for (int j = 0; j < x.dim(); j++)  {
          x.adjust(j, -step * v.get(j) / (Math.sqrt(c.get(j) + eps)));
        }
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
    System.out.printf("Adam Descent finished in %.2f seconds\n", (System.nanoTime() - startTime) / 1e9);
    return x;
  }
}
