package com.spbsu.ml.methods.greedyMergeOptimization;

import com.spbsu.commons.util.Holder;
import com.spbsu.commons.util.ThreadTools;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Created by noxoomo on 30/11/14.
 */

public class GreedyMergePick<Model, Comparator extends ModelComparators<Model>> {
  static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Greedy merge pick thread", -1);
  private MergeOptimization<Model> merger;

  public GreedyMergePick(MergeOptimization<Model> merger) {
    this.merger = merger;
  }

  public Model pick(List<Model> startModels, final Comparator comparator) {
    if (startModels.isEmpty())
      throw new IllegalArgumentException("Models list must be not empty");
    final TreeSet<Model> models = new TreeSet<>(comparator.regularizationComparator());
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
            try {
              Model merged = merger.merge(current, model);
              synchronized (best) {
                if (!best.filled() || comparator.targetComparator().compare(best.getValue(), merged) > 0)
                  best.setValue(merged);
              }
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

      if (comparator.scoreComparator().compare(best.getValue(), current) < 0) {
        models.add(best.getValue());
      }
    }
    return models.first();
  }
}


