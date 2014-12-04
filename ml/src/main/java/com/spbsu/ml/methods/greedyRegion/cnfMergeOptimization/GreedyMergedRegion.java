package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyMergeOptimization.GreedyMergePick;
import com.spbsu.ml.methods.greedyMergeOptimization.RegularizedLoss;
import com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors;
import com.spbsu.ml.models.CNF;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * Created by noxoomo on 30/11/14.
 */
public class GreedyMergedRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;
  private double lambda;

  public GreedyMergedRegion(BFGrid grid, double lambda) {
    this.lambda = lambda;
    this.grid = grid;
  }

  public GreedyMergedRegion(BFGrid grid) {
    this(grid, 2);
  }

  @Override
  public CNF fit(final VecDataSet learn, final Loss loss) {
    final List<CNF.Clause> clauses = new ArrayList<>(10);
    @SuppressWarnings("unchecked")
    CherryOptimizationSubsetMerger merger = new CherryOptimizationSubsetMerger(loss.statsFactory());
    int[] points = ArrayTools.sequence(0, learn.length());
    GreedyMergePick<CherryOptimizationSubset>
            pick = new GreedyMergePick<>(merger);
    double current = Double.POSITIVE_INFINITY;
    AdditiveStatistics inside = (AdditiveStatistics) loss.statsFactory().create();
    final AdditiveStatistics totalStat = (AdditiveStatistics) loss.statsFactory().create();
    for (int point : points) totalStat.append(point, 1);

    final BitSet[] used = new BitSet[grid.rows()];
    for (int i = 0; i < grid.rows(); ++i)
      used[i] = new BitSet(grid.row(i).size() + 1);

    while (true) {
      List<CherryOptimizationSubset> models = init(learn, points, used, loss);
      RegularizedLoss<CherryOptimizationSubset> regLoss = new RegularizedLoss<CherryOptimizationSubset>() {
        @Override
        public double target(CherryOptimizationSubset subset) {
          return loss.score(subset.stat);
        }

        @Override
        public double regularization(CherryOptimizationSubset subset) {
          double weight = weight(subset.stat);
          return -Math.log(weight+1) / subset.cardinality();
        }

        @Override
        public double score(CherryOptimizationSubset subset) {
          return loss.score(subset.stat) * (1 - lambda * regularization(subset));
        }
      };

      CherryOptimizationSubset best = pick.pick(models, regLoss);
      if (current <= score(best.stat)) {
        break;
      }
      clauses.add(best.clause);
      for (CNF.Condition condition : best.clause.conditions) {
        used[condition.feature].or(condition.used);
      }
      points = best.inside();
      inside = best.stat;
      current = score(best.stat);
    }

    CNF.Clause[] result = new CNF.Clause[clauses.size()];
    for (int i = 0; i < clauses.size(); ++i)
      result[i] = clauses.get(i);

    for (int i = 0; i < clauses.size(); ++i) {
      System.out.println("Clause " + i);
      for (CNF.Condition condition : result[i].conditions) {
        System.out.println("Feature " + condition.feature + " " + condition.used);
      }
    }

    System.out.println("Region weight " + AdditiveStatisticsExtractors.weight(inside));
    return new CNF(result, loss.bestIncrement(inside), grid);
  }


  public double score(AdditiveStatistics stats) {
    double sum = sum(stats);
    double weight = weight(stats);
    return weight > 1 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight + 1)) : 0;
  }

  static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Init CNF thread", -1);

  private List<CherryOptimizationSubset> init(final VecDataSet learn, final int[] points, final BitSet[] previouslyUsed, final Loss loss) {
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    int binsTotal = 0;
    for (int feature = 0; feature < grid.rows(); ++feature)
      binsTotal += grid.row(feature).size() > 1 ? grid.row(feature).size() + 1 : 0;

    final List<CherryOptimizationSubset> result = new ArrayList<>(binsTotal);
    final CountDownLatch latch = new CountDownLatch(binsTotal);
    for (int feature = 0; feature < grid.rows(); ++feature) {
      if (grid.row(feature).size() <= 1)
        continue;
      for (int bin = 0; bin <= grid.row(feature).size(); ++bin) {
        if (previouslyUsed[feature].get(bin)) {
          latch.countDown();
          continue;
        }
        final int finalFeature = feature;
        final int finalBin = bin;
        exec.submit(new Runnable() {
          @Override
          public void run() {
            final CNF.Condition[] conditions = new CNF.Condition[1];
            final BitSet used = new BitSet(grid.row(finalFeature).size() + 1);
            used.set(finalBin);
            conditions[0] = new CNF.Condition(finalFeature, used);
            final CNF.Clause clause = new CNF.Clause(grid, conditions);
            final CherryOptimizationSubset subset = new CherryOptimizationSubset(bds, (AdditiveStatistics) loss.statsFactory().create(), clause, points);
            synchronized (result) {
              result.add(subset);
            }
            latch.countDown();
          }
        });
      }
    }
    try {
      latch.await();
    } catch (InterruptedException e) {
      //skip
    }
    return result;
  }


}
