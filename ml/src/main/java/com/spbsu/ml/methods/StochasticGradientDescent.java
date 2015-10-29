package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.ROVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.DSSumFuncComposite;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;

/**
 * User: solar
 * Date: 01.06.15
 * Time: 12:49
 */
public class StochasticGradientDescent<Item> extends WeakListenerHolderImpl<Vec> implements Optimization<DSSumFuncComposite<Item>, DataSet<Item>,Item> {
  private final Executor executor;

  private final FastRandom rng;
  private final int couple;
  private final int T;
  private final double step;

  public StochasticGradientDescent(FastRandom rng, int couple, int T, double step) {
    this.rng = rng;
    this.couple = couple;
    this.T = T;
    this.step = step;
    executor = ThreadTools.createBGExecutor(StochasticGradientDescent.class.getName(), this.couple);
  }
  @Override
  public DSSumFuncComposite<Item>.Decision fit(DataSet<Item> learn, final DSSumFuncComposite<Item> target) {
    final Vec cursor = new ArrayVec(target.dim());
    init(cursor);
    final Vec[] coupleVec = new Vec[couple];
    final Vec gradient = new ArrayVec(target.dim());
    for (int t = 0; t < T; t++) {
      VecTools.fill(gradient, 0.);
      final CountDownLatch latch = new CountDownLatch(couple);
      for (int i = 0; i < couple; i++) {
//        final int nextItem = rng.nextInt(1000);
        final int nextItem = rng.nextInt(learn.length());
//        System.out.println("sample :" + learn.meta().owner().feature(0, nextItem) + " target: " + target.component(nextItem));
        final int finalI = i;
        executor.execute(new Runnable() {
          @Override
          public void run() {
            final Vec currentGrad = target.component(nextItem).gradient(new ROVec(cursor));
            coupleVec[finalI] = currentGrad;
            synchronized (gradient) {
              VecTools.append(gradient, currentGrad);
            }
            latch.countDown();
          }
        });
      }
      try {
        latch.await();
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
      VecTools.scale(gradient, 1. / couple);
      {
//        double meanCos = 0;
//        for (int i = 0; i < couple; i++) {
//          for (int j = 0; j < couple; j++) {
//            meanCos += VecTools.cosine(coupleVec[i], coupleVec[j]) / couple / couple;
//          }
//        }
////        System.out.println(gradient);
//        System.out.println(meanCos + " " + VecTools.norm(gradient));
      }
      normalizeGradient(gradient);
//      VecTools.scale(gradient, step * 100. / Math.sqrt(10000. + t));
      VecTools.scale(gradient, step);
      VecTools.append(cursor, gradient);
      invoke(new ROVec(cursor));
    }
    return target.decision(cursor);
  }

  public void init(Vec cursor) {
    VecTools.fillUniform(cursor, rng);
  }

  public void normalizeGradient(Vec grad) {
  }
}
