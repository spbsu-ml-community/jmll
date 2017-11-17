package com.expleague.ml;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.util.ThreadTools;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface BlockwiseFuncC1 extends FuncC1 {
  void gradient(Vec pointBlock, Vec result, int index);

  double value(Vec pointBlock, int index);
  double transformResultValue(double value);

  int blockSize();

  abstract class Stub extends FuncC1.Stub implements BlockwiseFuncC1 {
    public final Mx gradient(final Mx x) {
      final Mx result = VecTools.copy(x);
      final CountDownLatch latch = new CountDownLatch(ThreadTools.COMPUTE_UNITS);
      for (int t = 0; t < ThreadTools.COMPUTE_UNITS; t++) {
        final int finalT = t;
        ForkJoinPool.commonPool().execute(() -> {
          for (int i = finalT; i < x.rows(); i+= ThreadTools.COMPUTE_UNITS) {
            gradient(x.row(i), result.row(i), i);
          }
          latch.countDown();
        });
      }
      try {
        latch.await();
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
      return result;
    }

    @Override
    public final Mx gradient(final Vec x) {
      return gradient(x instanceof Mx ? (Mx)x : new VecBasedMx(blockSize(), x));
    }

    protected double value(final Mx blocks) {
      double result = 0.0;
      for (int i = 0; i < blocks.rows(); i ++) {
        result += value(blocks.row(i), i);
      }
      return result;
    }

    @Override
    public final double value(final Vec x) {
      final double value = value(x instanceof Mx ? (Mx) x : new VecBasedMx(blockSize(), x));
      return transformResultValue(value);
    }
  }
}
