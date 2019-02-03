package com.expleague.cuda.root.nn;


import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.cuda.KernelOperations;
import com.expleague.cuda.data.GPUVec;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.func.RegularizerFunc;
import com.expleague.ml.optimization.Optimize;

import java.util.Random;

/**
 * Created by hrundelb on 25.09.17.
 */
public class SAGADescentGPU implements Optimize<FuncEnsemble<? extends FuncC1>> {
  private final double step;
  private final int maxIter;
  private final Random random;
  private final int threadCount;
  private long time;

  public SAGADescentGPU(final double step, final int maxIter, final Random random, final int threadCount) {
    this.step = step;
    this.maxIter = maxIter;
    this.random = random;
    this.threadCount = threadCount;
  }

  @Override
  public Vec optimize(FuncEnsemble<? extends FuncC1> func, RegularizerFunc reg, Vec x0) {
    return null;
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

    GPUVec x;
    if (threadCount == 1) {
      x = new GPUVec(x0.toArray());
    } else {
      throw new UnsupportedOperationException("No multithread support yet");
    }

    final Vec[] lastGrad = new GPUVec[sumFuncs.size()];
    final Vec totalGrad = new GPUVec(x.dim());
    final Vec tmp = new ArrayVec(x.dim());

    for (int i = 0; i < sumFuncs.size(); i++) {
      lastGrad[i] = sumFuncs.models[i].gradient(x);
      VecTools.append(totalGrad, lastGrad[i]);
      final VecIterator iterator = new ArrayVec(lastGrad[i].toArray()).nonZeroes();
      while (iterator.advance()) {
        tmp.adjust(iterator.index(), 1);
      }
    }
    for (int i = 0; i < x.dim(); i++) {
      tmp.set(i, 1.0 * sumFuncs.size() / tmp.get(i));
    }
    final Vec gradCoordinateInverseFreq = new GPUVec(tmp.toArray());

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
    time = System.currentTimeMillis();
    for (int iter = 1; iter < maxIter / threadCount; iter++) {
      final int i = random.nextInt(sumFuncs.size());

      Vec grad = VecTools.scale(lastGrad[i], -1);

      lastGrad[i] = sumFuncs.models[i].gradient(x);
      VecTools.append(grad, lastGrad[i]);

      VecTools.incscale(x, grad, -step);

      KernelOperations.fVectorKernel1((GPUVec)lastGrad[i], (GPUVec)gradCoordinateInverseFreq,
          (GPUVec) totalGrad, (float)step, sumFuncs.size(), (GPUVec) x);

      VecTools.append(totalGrad, grad); // total += new grad - old grad


      if (iter % (5 * sumFuncs.size()) == 0) {
        final double curError = sumFuncs.value(x) / sumFuncs.size();
        final long newTime = System.currentTimeMillis();
        System.out.printf("thread %d, iteration %d: new=%.6f old=%.6f time=%dms\n", Thread
            .currentThread().getId(), iter, curError, error, newTime - time);
        time = newTime;
        if (curError > error) {
          break;
        }
        error = curError;
      }
    }
  }
}