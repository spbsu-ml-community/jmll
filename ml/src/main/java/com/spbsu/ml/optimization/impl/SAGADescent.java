package com.spbsu.ml.optimization.impl;

import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.optimization.Optimize;

import java.util.Random;

public class SAGADescent implements Optimize<FuncEnsemble<? extends FuncC1>> {
  private final double step;
  private final int maxIter;
  private final Random random;
  private final int threadCount;

  public SAGADescent(final double step, final int maxIter, final Random random, final int threadCount) {
    this.step = step;
    this.maxIter = maxIter;
    this.random = random;
    this.threadCount = threadCount;
  }

  @Override
  public Vec optimize(final FuncEnsemble<? extends FuncC1> sumFuncs) {
    final Vec x = new ArrayVec(sumFuncs.dim());
    for (int i = 0; i < sumFuncs.dim(); i++) {
      x.set(i, random.nextGaussian());
    }
    return optimize(sumFuncs, x);
  }

  @Override
  public Vec optimize(final FuncEnsemble<? extends FuncC1> sumFuncs, final Vec x0) {
    final long startTime = System.nanoTime();

    Vec x;
    if (threadCount == 1) {
      x = VecTools.copy(x0);
    } else {
//      x = new AtomicArrayVec(x0.dim());
//      for (int i = 0; i < x0.dim(); i++) {
//        x.set(i, x0.get(i));
//      } TODO
      throw new UnsupportedOperationException("No multithread support yet");
    }

    final Vec[] lastGrad = new Vec[sumFuncs.size()];
    final Vec totalGrad = new ArrayVec(x.dim());
    final Vec gradCoordinateInverseFreq = new ArrayVec(x.dim());

    for (int i = 0; i < sumFuncs.size(); i++) {
      lastGrad[i] = sumFuncs.models[i].gradient(x);
      VecTools.append(totalGrad, lastGrad[i]);
      final VecIterator iterator = lastGrad[i].nonZeroes();
      while (iterator.advance()) {
        gradCoordinateInverseFreq.adjust(iterator.index(), 1);
      }
    }
    for (int i = 0; i < x.dim(); i++) {
      gradCoordinateInverseFreq.set(i, 1.0 * sumFuncs.size() / gradCoordinateInverseFreq.get(i));
    }

    if (threadCount == 1) {
      run(sumFuncs, gradCoordinateInverseFreq, x, lastGrad, totalGrad);
    } else {
      Thread[] thread = new Thread[threadCount];
      for (int i = 0; i < threadCount; i++) {
        thread[i] = new Thread(() -> run(sumFuncs, gradCoordinateInverseFreq, x, lastGrad, totalGrad));
        thread[i].start();
      }
      for (int i = 0; i < threadCount; i++) {
        try {
          thread[i].join();
        }
        catch (InterruptedException e) {
          e.printStackTrace();
        }
      }
    }
    System.out.println(sumFuncs.value(x) / sumFuncs.size());
    final long curTime = System.nanoTime();
    System.out.printf("SAGA Descent finished in %.2f seconds\n", (curTime - startTime) / 1e9);
    return x;
  }

  private void run(final FuncEnsemble<? extends FuncC1> sumFuncs, final Vec gradCoordinateInverseFreq, Vec x,
                   final Vec[] lastGrad, final Vec totalGrad) {
    double error = sumFuncs.value(x) / sumFuncs.size();
    for (int iter = 1; iter < maxIter / threadCount; iter++) {
      final int i = random.nextInt(sumFuncs.size());

      Vec grad = VecTools.scale(lastGrad[i], -1);
      lastGrad[i] = sumFuncs.models[i].gradient(x);
      VecTools.append(grad, lastGrad[i]);

      VecTools.incscale(x, grad, -step);
      final VecIterator iterator = lastGrad[i].nonZeroes();
      while (iterator.advance()) {
        final int index = iterator.index();
        x.adjust(index, -step * gradCoordinateInverseFreq.get(index) * totalGrad.get(index) / sumFuncs.size());
      }

      VecTools.append(totalGrad, grad); // total += new grad - old grad


      if (iter % (5 * sumFuncs.size()) == 0) {
        final double curError = sumFuncs.value(x) / sumFuncs.size();
        System.out.printf("thread %d, iteration %d: new=%.6f old=%.6f\n", Thread.currentThread().getId(), iter, curError, error);
        if (curError > error) {
          break;
        }
        error = curError;
      }
    }
  }
}