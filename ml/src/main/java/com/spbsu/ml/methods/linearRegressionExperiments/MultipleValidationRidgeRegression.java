package com.spbsu.ml.methods.linearRegressionExperiments;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.func.Linear;
import com.spbsu.ml.loss.L2;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static com.spbsu.commons.math.vectors.VecTools.l2;

/**
 * Created by noxoomo on 12/06/15.
 */
public class MultipleValidationRidgeRegression {
  final double minLambda = 1e-12;
  private static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("ridge regressions", -1);

  public Linear[] fit(final VecDataSet[] learn,
                      final L2[] loss,
                      final VecDataSet[] validationDs,
                      final L2[] valLoss) {
    if (learn.length != loss.length
      || loss.length != validationDs.length
      || validationDs.length != valLoss.length)
      throw new IllegalArgumentException("losses count ≠ ds count");

    final boolean[] empty = new boolean[learn.length];
    final int[] map = new int[learn.length];


    int featureCount = 0;
    final int effectiveCount;
    Linear zeroWeight = null;
    {
      int ind = 0;
      for (int i = 0; i < learn.length; ++i) {
        if (loss[i] == null || loss[i].dim()  < 2*learn[i].xdim()) {
          empty[i] = true;
        } else {
          featureCount = learn[i].xdim();
          map[i] = ind++;
        }
      }
      effectiveCount = ind;
    }

    final RidgeRegressionCache[] regressions = new RidgeRegressionCache[effectiveCount];

    {
      final CountDownLatch latch = new CountDownLatch(learn.length);
      for (int i = 0; i < learn.length; ++i) {
        final int ind = i;
        exec.submit(new Runnable() {
          @Override
          public void run() {
            if (!empty[ind]) {
              final RidgeRegressionCache cache = new RidgeRegressionCache(learn[ind], loss[ind]);
              regressions[map[ind]] = cache;
            }
            latch.countDown();
          }
        });
      }
      try {
        latch.await();
      } catch (InterruptedException e) {
        e.printStackTrace();
      }

    }


//    double lambda = 1.0;
    final double lambdas[] = new double[effectiveCount];
    final boolean stopped[] = new boolean[effectiveCount];
    Arrays.fill(lambdas,1.0);

//    double bestScore = Double.POSITIVE_INFINITY;
    final double[] bestScores = new double[effectiveCount];
    Arrays.fill(bestScores,Double.POSITIVE_INFINITY);
    final Linear[] result;
    {
      final double[] scores = new double[effectiveCount];
      final Linear[] currentResult = new Linear[effectiveCount];

//      while (lambda > minLambda) {
      while (true) {
        Arrays.fill(scores, 0);
        final CountDownLatch latch = new CountDownLatch(learn.length);

//        final double fLambda = lambda;

        for (int i = 0; i < learn.length; ++i) {
          final int ind = i;
          exec.submit(new Runnable() {
            @Override
            public void run() {
              final int index = map[ind];
              if (!stopped[index]) {
                final Linear model = regressions[index].fit(lambdas[index]);
                currentResult[index] = model;
                final Mx data = validationDs[ind].data();
                if (data.rows() != 0) {
                  Vec predictions = model.transAll(data);
                  scores[index] = l2(predictions, valLoss[ind].target) / data.rows();
                }
              }
              latch.countDown();
            }
          });
        }

        try {
          latch.await();
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
        boolean updated = false;
        for (int i=0; i < lambdas.length;++i) {
          if (!stopped[i] && scores[i] < bestScores[i] && lambdas[i] > minLambda) {
            updated  = true;
            lambdas[i] /= 2;
            bestScores[i] = scores[i];
          } else {
            stopped[i] = true;
          }
        }
        if (!updated) {
          break;
        }
//        double score = scores[ArrayTools.max(scores)];
//        if (score > bestScore) {
//          break;
//        }
//        bestScore = score;
//        lambda /= 2;
      }
      result = currentResult;
    }

    Linear[] totalResult = new Linear[empty.length];
    int ind = 0;
    for (int i = 0; i < empty.length; ++i) {
      if (empty[i]) {
        if (zeroWeight == null) {
          zeroWeight = new Linear(new double[featureCount]);
        }
        totalResult[i] = zeroWeight;
      } else {
        totalResult[i] = result[ind++];
      }
    }
    return totalResult;
  }

}
