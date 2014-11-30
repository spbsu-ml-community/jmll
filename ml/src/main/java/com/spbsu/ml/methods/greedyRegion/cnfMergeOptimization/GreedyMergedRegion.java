package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.greedyMergeOptimization.GreedyMergePick;
import com.spbsu.ml.methods.greedyMergeOptimization.RegularizedLoss;
import com.spbsu.ml.methods.greedyMergeOptimization.RegularizedLossComparator;
import com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors;
import com.spbsu.ml.models.CNF;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * Created by noxoomo on 30/11/14.
 */
public class GreedyMergedRegion<Loss extends StatBasedLoss<AdditiveStatistics>> extends VecOptimization.Stub<Loss> {
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
    CherryOptimizationSubsetMerger merger = new CherryOptimizationSubsetMerger(loss);
    int[] points = ArrayTools.sequence(0, learn.length());
    GreedyMergePick<CherryOptimizationSubset, RegularizedLossComparator<CherryOptimizationSubset, RegularizedLoss<CherryOptimizationSubset>>>
            pick = new GreedyMergePick<>(merger);
//    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    double current = Double.POSITIVE_INFINITY;
    AdditiveStatistics inside = loss.statsFactory().create();

    final BitSet[] used = new BitSet[grid.rows()];
    for (int i = 0; i < grid.rows(); ++i)
      used[i] = new BitSet(grid.row(i).size() + 1);

    while (true) {
//      final ModelComplexityCalcer calcer = new ModelComplexityCalcer(bds, points, used);
      List<CherryOptimizationSubset> models = init(learn, points, loss);

      RegularizedLoss<CherryOptimizationSubset> regLoss = new RegularizedLoss<CherryOptimizationSubset>() {

        @Override
        public double target(CherryOptimizationSubset subset) {
          return loss.score(subset.stat);
        }

        @Override
        public double regularization(CherryOptimizationSubset subset) {
//          if (!subset.isRegularizationKnown) {
//            subset.regularization = calcer.calculate(subset.layer);
//            subset.isRegularizationKnown = true;
//          }
//          return subset.regularization;
          double weight = weight(subset.stat);
          return -lambda * Math.log(weight + 1);
        }

        @Override
        public double score(CherryOptimizationSubset subset) {
          return loss.score(subset.stat) * (1 - lambda * regularization(subset));
        }
      };

      RegularizedLossComparator<CherryOptimizationSubset, RegularizedLoss<CherryOptimizationSubset>> comparator = new RegularizedLossComparator<>(regLoss);
      CherryOptimizationSubset best = pick.pick(models, comparator);
      if (current <= score(best.stat)) {
        break;
      }
      clauses.add(best.clause);
      for (CNF.Condition condition : best.clause.conditions) {
        used[condition.feature].or(condition.used);
      }
      points = best.inside;
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
    return weight > 1 ? (-sum * sum / weight) * weight * (weight - 2) / (weight * weight - 3 * weight + 1) * (1 + 2 * Math.log(weight)) : 0;
  }


  private List<CherryOptimizationSubset> init(VecDataSet learn, int[] points, Loss loss) {
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    List<CherryOptimizationSubset> result = new ArrayList<>();
    for (int feature = 0; feature < grid.rows(); ++feature) {
      if (grid.row(feature).size() <= 1)
        continue;
      for (int bin = 0; bin <= grid.row(feature).size(); ++bin) {
        CNF.Condition[] conditions = new CNF.Condition[1];
        BitSet used = new BitSet(grid.row(feature).size() + 1);
        used.set(bin);
        conditions[0] = new CNF.Condition(feature, used);
        CNF.Clause clause = new CNF.Clause(grid, conditions);
        CherryOptimizationSubset subset = new CherryOptimizationSubset(bds, (AdditiveStatistics) loss.statsFactory().create(), clause, points);
        result.add(subset);
      }
    }
    return result;
  }

  class ModelComplexityCalcer {
    private final BFGrid grid;
    private final int[][] base;
    BitSet[] used;
    public double total;

    public ModelComplexityCalcer(BinarizedDataSet bds, int[] points, BitSet[] used) {
      this.grid = bds.grid();
      this.used = used;
      base = new int[grid.rows()][];
      {
        for (int feature = 0; feature < grid.rows(); feature++) {
          base[feature] = new int[grid.row(feature).size() + 1];
          final byte[] bin = bds.bins(feature);
          for (int j = 0; j < points.length; j++) {
            base[feature][bin[points[j]]]++;
          }
        }
      }
      total = 0;
      for (int bin = 0; bin <= grid.row(0).size(); ++bin) {
        total += base[0][bin];
      }
    }

    public double calculate(CNF.Clause clause) {
      double reg = Double.POSITIVE_INFINITY;
      for (CNF.Condition condition : clause.conditions) {
        double information = 0;
        double count = 0;
        int f = condition.feature;
        boolean current = false;
        for (int bin = 0; bin <= grid.row(f).size(); ++bin) {
          if (condition.used.get(bin) && used[f].get(bin)) { //don't use features, which was used on previous levels
            return Double.POSITIVE_INFINITY;
          }
          if (condition.used.get(bin) == current) {
            count += base[f][bin];
          } else {
//            information += count > 0 ? count * Math.log(count) : 0;
            information += Math.log(count + 1);
            current = condition.used.get(bin);
            count = base[f][bin];
          }
          if (current && base[f][bin] == 0)
            return Double.POSITIVE_INFINITY;
        }
        information += count > 0 ? count * Math.log(count) : 0;
//        information +=  Math.log(count+1);
        information /= total;
//        double entropy =  - information;
//        reg += information;
        reg = Math.min(information, reg);
      }
      return -reg / clause.conditions.length;//reg / layer.conditions.length;

    }
  }

}
