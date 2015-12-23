package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 10/06/15.
 */
public class RidgeRegression implements VecOptimization<L2> {
  final double alpha;

  final double multiplyArrayVec(Vec left, Vec right) {
    assert (left.dim() == right.dim());
    int size = (left.dim() / 4) * 4;
    double result = 0;
    for (int i = 0; i < size; i += 4) {
      final double la = left.get(i);
      final double lb = left.get(i + 1);
      final double lc = left.get(i + 2);
      final double ld = left.get(i + 3);
      final double ra = right.get(i);
      final double rb = right.get(i + 1);
      final double rc = right.get(i + 2);
      final double rd = right.get(i + 3);

      final double dpa = la * ra;
      final double dpb = lb * rb;
      final double dpc = lc * rc;
      final double dpd = ld * rd;

      result += (dpa + dpb) + (dpc + dpd);
    }
    for (int i = size; i < left.dim(); ++i) {
      result += left.get(i) * right.get(i);
    }
    return result;
  }

  public RidgeRegression(double alpha) {
    this.alpha = alpha;
  }

  @Override
  public Linear fit(VecDataSet learn, L2 l2) {
    Vec target = l2.target;
    Mx data = learn.data();
    return new Linear(fit(data, target));
  }

  static final ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Ridge dot-products thread", -1);
  ;

  final public Vec fit(final Mx data, final Vec target) {
    final Mx cov = new VecBasedMx(data.columns(), data.columns());
    final Vec covTargetWithFeatures = new ArrayVec(data.columns());

    final CountDownLatch latch = new CountDownLatch(data.columns());

    for (int col = 0; col < data.columns(); ++col) {
      final int i = col;
      exec.submit((Runnable) () -> {
        final Vec feature = data.col(i);
        cov.set(i, i, multiplyArrayVec(feature, feature));
        cov.adjust(i, i, alpha);
        covTargetWithFeatures.set(i, multiplyArrayVec(feature, target));
        for (int j = i + 1; j < data.columns(); ++j) {
          final double val = multiplyArrayVec(feature, data.col(j));
          cov.set(i, j, val);
          cov.set(j, i, val);
        }
        latch.countDown();
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      //
    }

    Mx invCov = MxTools.inverse(cov);
    return MxTools.multiply(invCov, covTargetWithFeatures);
  }
}
