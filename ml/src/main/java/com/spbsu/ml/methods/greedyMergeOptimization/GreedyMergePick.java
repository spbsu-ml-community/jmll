package com.spbsu.ml.methods.greedyMergeOptimization;

import com.spbsu.commons.util.ThreadTools;

import java.util.Collections;
import java.util.List;
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

  public Model pick(List<Model> startModels, Comparator comparator) {
    List<Model> models = startModels;
    Model current = models.get(0);
    for (int k = 1; k < models.size(); ++k) {
      if (comparator.targetComparator().compare(models.get(k), current) < 0) {
        current = models.get(k);
      }
    }

    while (true) {
      //lower index means model is better (from regularization point of view)
      Collections.sort(models, comparator.regularizationComparator());
      boolean updated = false;
//      for (int i = models.size() - 1; i > 0.8*models.size(); --i) {
      for (int i = models.size() - 1; i > 0; --i) {
        final Model last = models.get(i);
        models.remove(i);

        @SuppressWarnings("unchecked")
        final Model[] result = (Model[]) new Object[i];

        final CountDownLatch latch = new CountDownLatch(result.length);
        for (int j = 0; j < i; ++j) {
          final int index = j;
          final Model toMerge = models.get(index);
          exec.submit(new Runnable() {
            @Override
            public void run() {
              Model merged = merger.merge(last, toMerge);
              result[index] = merged;
              latch.countDown();
            }
          });
        }

        try {
          latch.await();
        } catch (InterruptedException e) {
          //skip
        }


        Model best = result[0];
        int bestIndex = 0;
        for (int k = 1; k < result.length; ++k) {
          if (comparator.targetComparator().compare(result[k], best) < 0) {
            best = result[k];
            bestIndex = k;
          }
          if (comparator.scoreComparator().compare(result[k], current) < 0) {
            current = result[k];
            updated = true;
          }
        }

        if (comparator.targetComparator().compare(best, last) < 0 && comparator.targetComparator().compare(best, models.get(bestIndex)) < 0) {
          //newModels.add(best);
          models.set(bestIndex, best);
          break;
        }
      }
      if (!updated) {
        break;
      }
    }
    return current;
  }
}


