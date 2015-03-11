package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.CherryPick;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.CNF;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDCherryRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  protected final BFGrid grid;

  public GreedyTDCherryRegion(final BFGrid grid) {
    this.grid = grid;
  }

  @Override
  public CNF fit(final VecDataSet learn, final Loss loss) {
    final List<CNF.Clause> conditions = new ArrayList<>(100);
    final List<CherryPick.ClauseComplexity> complexities = new ArrayList<>(100);
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    final CherryPick pick = new CherryPick(bds, loss.statsFactory());
    int[] points = ArrayTools.sequence(0, learn.length());

    double currentScore = Double.NEGATIVE_INFINITY;
    AdditiveStatistics inside = (AdditiveStatistics) loss.statsFactory().create();
    while (true) {
      final CherryPick.Result result = pick.fit(points, new Evaluator<AdditiveStatistics>() {
        @Override
        public double value(AdditiveStatistics additiveStatistics) {
          return -loss.score(additiveStatistics);
        }
      });


      complexities.add(result.complexity);
      final double newScore = score(complexities);
      if (currentScore >=  newScore) {
        break;
      }
      System.out.println("\nAdded clause " + result.clause);

      points = result.points;
      conditions.add(result.clause);
      inside = result.complexity.inside;
      currentScore = newScore;
    }

    return new CNF(conditions.toArray(new CNF.Clause[conditions.size()]),loss.bestIncrement(inside),grid);
  }

  private double score(List<CherryPick.ClauseComplexity> complexities) {
    final AdditiveStatistics inside = complexities.get(complexities.size()-1).inside;
    final AdditiveStatistics total = complexities.get(complexities.size()-1).total;
    int complexity = 0;
    for (CherryPick.ClauseComplexity clause : complexities) {
      complexity += clause.complexity;
    }
//    double best = 0;
//    for (CherryPick.ClauseComplexity clause : complexities) {
//      complexity += clause.complexity;
//      final double reg = clause.logsSum;
//      final double totalWeight = weight(clause.total);
//      final double bestWeight= weight(clause.total) / (clause.complexity+1);
//      double bestReg = (clause.complexity+1) * bestWeight * Math.log(1.0 /(clause.complexity+1));
//      best += reg / bestReg;
//    }
//    best /= complexities.size();
//    complexity //= complexities.size();
    double s = sum(inside);
    double w = weight(inside);
    double reg = Math.log(w) / (complexity + 2);
    return w > 2 ? (s * s / w) * w * (w - 2) / (w * w - 3 * w + 1) * (1 +16*reg) : 0;
  }
}


