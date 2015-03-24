package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.math.MathTools;
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
import static com.spbsu.ml.methods.greedyRegion.GreedyTDRegionNonStochasticProbs.sum2;

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
    AdditiveStatistics outside = (AdditiveStatistics) loss.statsFactory().create();
    AdditiveStatistics total = (AdditiveStatistics) loss.statsFactory().create();
    for (int i =0 ; i < learn.length();++i)
      total.append(i,1);
    double currentReg = 0;
    while (true) {
      final CherryPick.Result result = pick.fit(points, outside, currentReg,loss);
      complexities.add(result.complexity);
      AdditiveStatistics out = ((AdditiveStatistics) loss.statsFactory().create()).append(total).remove(result.complexity.inside);
      final double newScore = (loss.score(result.complexity.inside) + loss.score(out)) / (currentReg + result.complexity.reg);

      //score(complexities, total);//loss.score(result.complexity.inside);// + loss.value(out);
      if (currentScore >= newScore) {
        break;
      }
      System.out.println("\nAdded clause " + result.clause);
      points = result.points;
      conditions.add(result.clause);
      inside = result.complexity.inside;
      currentReg += result.complexity.reg;
      outside = out;
      currentScore = newScore;
    }

//    System.out.println("\nCNF weight inside: " + weight(inside) + "; weight outside: "+ (weight(total) - weight(inside))) ;
    return new CNF(conditions.toArray(new CNF.Clause[conditions.size()]), loss.bestIncrement(inside) , loss.bestIncrement(outside), grid);
  }

  private double score(List<CherryPick.ClauseComplexity> complexities, AdditiveStatistics total) {
    final AdditiveStatistics inside = complexities.get(complexities.size() - 1).inside;
    final AdditiveStatistics outside = complexities.get(complexities.size() - 1).outside;
    final double totalWeight = weight(complexities.get(0).inside) + weight(complexities.get(0).outside);
    int complexity = 0;
    for (CherryPick.ClauseComplexity clause : complexities) {
      complexity += clause.complexity;
    }

    double currentReg = 0.0;
    double currentRegOut = 0.0;
    final double ideaSplit = Math.log(0.5);
    for (CherryPick.ClauseComplexity clause : complexities) {
//      final double in = weight(clause.inside);
//      final double out = weight(clause.outside);
//      if (in == 0 || out == 0) {
//        return Double.NEGATIVE_INFINITY;
//      }
//      double p = in / (in + out);
      currentReg += clause.reg;// (2*clause.complexity -ideaSplit + p * Math.log(p) + (1-p) * Math.log(1-p));
    }
//    for (CherryPick.ClauseComplexity clause : complexities) {
//      currentReg += clause.reg;
//    }
//    currentReg /= complexities.size();
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
    final double score;
    {
      double sumIn = sum(inside);
      double sum2In = sum2(inside);
      double sum2Out = sum2(total) - sum2In;
      double wIn = weight(inside);
      double wOut = weight(total) - wIn;
      if (wIn < 2 || wOut < 2) {
        return Double.NEGATIVE_INFINITY;
      }


      double var = wOut * (sum2Out / (wOut - 1))
              + MathTools.sqr(wIn / (wIn - 1)) * (sum2In - sumIn * sumIn / wIn);
      score = -var * (1 + 0.01*currentReg);
    }
    return score;
  }
}


