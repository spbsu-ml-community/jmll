package com.spbsu.ml.methods.greedyMergeOptimization;

import com.spbsu.commons.util.Holder;
import com.spbsu.commons.util.ThreadTools;

import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 30/11/14.
 */

public class GreedyMergePick<Model> {
  static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Greedy merge pick thread", -1);
  private MergeOptimization<Model> merger;

  public GreedyMergePick(MergeOptimization<Model> merger) {
    this.merger = merger;
  }

  public Model pick(List<Model> startModels, final RegularizedLoss<Model> loss) {
    if (startModels.isEmpty())
      throw new IllegalArgumentException("Models list must be not empty");

    final Comparator<Model> comparator = new Comparator<Model>() {
      @Override
      public int compare(Model left, Model right) {
        return Double.compare(loss.regularization(left), loss.regularization(right));
      }
    };

    final TreeSet<Model> models = new TreeSet<>(comparator);
    models.addAll(startModels);
    while (models.size() > 1) {
      final Model current;
      {
        final Iterator<Model> iterator = models.descendingIterator();
        current = iterator.next();
        iterator.remove();
      }
      final CountDownLatch latch = new CountDownLatch(models.size());
      final Holder<Model> best = new Holder<>(current);
      for (final Model model : models) {
        exec.submit(new Runnable() {
          @Override
          public void run() {
            Model merged = merger.merge(current, model);
            try {
              synchronized (best) {
                if (!best.filled() || loss.target(merged) + 1e-9 < loss.target(best.getValue()))
                  best.setValue(merged);
              }
            } finally {
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

      if (loss.score(best.getValue()) + 1e-9 < loss.score(current)) {
        models.add(best.getValue());
      }
    }
    return models.first();
  }
}


