package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.ThreadTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyMergeOptimization.GreedyMergePick;
import com.spbsu.ml.methods.greedyMergeOptimization.RegularizedLoss;
import com.spbsu.ml.models.CNF;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;
import static java.lang.Math.log;

/**
 * User: noxoomo
 * Date: 30/11/14
 * Time: 15:31
 */
public class GreedyMergedRegion<Loss extends StatBasedLoss<AdditiveStatistics>> extends VecOptimization.Stub<Loss> {
  public static final double CARDINALITY_FACTOR = 2;
  protected final BFGrid grid;
  private final double lambda;

  public GreedyMergedRegion(final BFGrid grid, final double lambda) {
    this.lambda = lambda;
    this.grid = grid;
  }

  public GreedyMergedRegion(final BFGrid grid) {
    this(grid, 2);
  }

  @Override
  public CNF fit(final VecDataSet learn, final Loss loss) {
    final List<CNF.Clause> clauses = new ArrayList<>(10);
    final CherryOptimizationSubsetMerger merger = new CherryOptimizationSubsetMerger(loss.statsFactory());
    final GreedyMergePick<CherryOptimizationSubset> pick = new GreedyMergePick<>(merger);
    int[] points = loss instanceof WeightedLoss ? ((WeightedLoss) loss).points(): ArrayTools.sequence(0, learn.length());
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    CherryOptimizationSubset last = new CherryOptimizationSubset(bds, loss.statsFactory(), new CNF.Clause(grid), points, 0);
    final double totalPower = last.power();
    final RegularizedLoss<CherryOptimizationSubset> regLoss = new RegularizedLoss<CherryOptimizationSubset>() {
      @Override
      public double target(CherryOptimizationSubset subset) {
        return -loss.score(subset.stat);
      }

      @Override
      public double regularization(CherryOptimizationSubset subset) {
        final double cardinalityDiscount = CARDINALITY_FACTOR / (subset.cardinality() + CARDINALITY_FACTOR - 1);
        final double regularization = (1 + lambda * log(subset.power() + 1));
        return cardinalityDiscount * regularization;
      }

      @Override
      public double score(CherryOptimizationSubset subset) {
        final double score = loss.score(subset.stat);
        return score * regularization(subset);
      }
    };

    double score = regLoss.score(last);
    while (true) {
      final List<CherryOptimizationSubset> models = init(bds, points, loss, last.cardinality());
      final CherryOptimizationSubset best = pick.pick(models, regLoss);

      System.out.print("\tClause " + clauses.size() + " score: " + regLoss.score(best) + " target: " + regLoss.target(best) + best.clause.toString());
      if (score - regLoss.score(best) < MathTools.EPSILON)
        break;
      System.out.println(" accepted");

      clauses.add(best.clause);
      points = best.inside();
      score = regLoss.score(best);
      last = best;
    }

    System.out.println(" rejected");

    System.out.println("Region weight: " + last.power() + " score: " + score + " target: " + regLoss.target(last));
    return new CNF(clauses.toArray(new CNF.Clause[clauses.size()]), loss.bestIncrement(last.stat), grid);
  }


  static ThreadPoolExecutor exec = ThreadTools.createBGExecutor("Init CNF thread", -1);

  private List<CherryOptimizationSubset> init(final BinarizedDataSet bds, final int[] points, final Loss loss, final double cardinality) {
    int binsTotal = 0;
    for (int feature = 0; feature < grid.rows(); ++feature)
      binsTotal += grid.row(feature).size() > 1 ? grid.row(feature).size() + 1 : 0;

    final List<CherryOptimizationSubset> result = new ArrayList<>(binsTotal);
    final CountDownLatch latch = new CountDownLatch(binsTotal);
    for (int feature = 0; feature < grid.rows(); ++feature) {
      if (grid.row(feature).size() <= 1)
        continue;
      for (int bin = 0; bin <= grid.row(feature).size(); ++bin) {
        final BFGrid.BFRow row = grid.row(feature);
        final BitSet used = new BitSet(row.size() + 1);
        used.set(bin);
        exec.submit(new Runnable() {
          @Override
          public void run() {
            final CNF.Condition[] conditions = new CNF.Condition[1];
            conditions[0] = new CNF.Condition(row, used);
            final CNF.Clause clause = new CNF.Clause(grid, conditions);
            final CherryOptimizationSubset subset = new CherryOptimizationSubset(bds, loss.statsFactory(), clause, points, cardinality);
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
