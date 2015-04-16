package com.spbsu.ml.methods.greedyMergeOptimization;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.util.BestHolder;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization.CherryOptimizationSubset;

import java.text.NumberFormat;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 30/11/14.
 */

public class GreedyMergePick<Model extends CherryOptimizationSubset> {
  static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Greedy merge pick thread", -1);
  private final MergeOptimization<Model> merger;

  public GreedyMergePick(final MergeOptimization<Model> merger) {
    this.merger = merger;
  }

  public Model pick(final List<Model> startModels, final RegularizedLoss<Model> loss) {
    final NumberFormat pp = MathTools.numberFormatter();
    if (startModels.isEmpty())
      throw new IllegalArgumentException("Models list must be not empty");

    final Comparator<Model> comparator = new Comparator<Model>() {
      @Override
      public int compare(final Model left, final Model right) {
        final int cmp = Double.compare(loss.score(left), loss.score(right));
        return cmp != 0 ? cmp : Integer.compare(left.index(), right.index());
      }
    };

    final TreeSet<Model> models = new TreeSet<>(comparator);
    models.addAll(startModels);
    while (models.size() > 1) {
      foo(loss, pp, models);
    }
    return models.first();
  }

  private void foo(final RegularizedLoss<Model> loss, final NumberFormat pp, final TreeSet<Model> models) {
    final Model current;
    {
      final Iterator<Model> iterator = models.descendingIterator();
      current = iterator.next();
      iterator.remove();
    }
    final CountDownLatch latch = new CountDownLatch(models.size());
    final double currentTarget = loss.target(current);
    final double currentReg = loss.regularization(current);
    final BestHolder<Model> bestHolder = new BestHolder<>();
//    System.out.print(current.toString() + " score: " + pp.format(currentScore));
    for (final Model model : models) {
      exec.submit(new Runnable() {
        @Override
        public void run() {
          try {
            final Model merged = merger.merge(current, model);
            if (merged.power() > model.power() && merged.power() > current.power()) {
              final double mergedTarget = loss.target(merged);
              final double mergedReg = loss.regularization(merged);
              final double modelTarget = loss.target(model);
              final double modelReg = loss.regularization(model);
              final double gain = (merged.power() * ((modelTarget + currentTarget) / (model.power() + current.power())) - mergedTarget);
              bestHolder.update(merged, gain);
            }
          }
          catch(Throwable th) {
            th.printStackTrace();
          }
          finally {
            latch.countDown();
          }
        }
      });
    }

    try {
      latch.await();
    } catch (InterruptedException e) {
      //skip
    }
    final Model best = bestHolder.getValue();
    if (bestHolder.getScore() > 0) {
      models.add(best);
    }
//    else System.out.print(" WASTED ");
//    if (bestHolder.filled())
//      System.out.println(" -> " + best.toString() + " score: " + pp.format(loss.score(best)) + " gain: " + bestHolder.getScore());
//    else
//      System.out.println();
  }
}


