package com.expleague.ml.methods.greedyRegion.cherry;

import com.expleague.ml.data.cherry.CherryLoss;
import com.expleague.ml.data.cherry.CherryPointsHolder;
import com.expleague.ml.loss.StatBasedLoss;
import com.expleague.ml.methods.greedyRegion.AdditiveStatisticsExtractors;
import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.ml.BFGrid;

public class OutLoss2<Subset extends CherryPointsHolder, Loss extends StatBasedLoss<AdditiveStatistics>> extends CherryLoss {
  private Subset subset;
  private Loss loss;
  private int complexity = 1;
  private int minBinSize = 150;


  OutLoss2(Subset subset, Loss loss) {
    this.subset = subset;
    this.loss = loss;
  }

  @Override
  public double score(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out) {
    if (start == 0 && end == feature.size())
      return Double.NEGATIVE_INFINITY;
    AdditiveStatistics currentInside = subset.inside();
    if (AdditiveStatisticsExtractors.weight(currentInside) != 0) {

      final int borders = borders(feature, start, end);
      final double c =  insideIncrement();
      final double addedWeight = AdditiveStatisticsExtractors.weight(added);
      final double addedSum = AdditiveStatisticsExtractors.sum(added);
      final double addedSum2 = AdditiveStatisticsExtractors.sum2(added);
      final double outSum = AdditiveStatisticsExtractors.sum(out);
      final double outSum2 = AdditiveStatisticsExtractors.sum2(out);
      final double outWeight = AdditiveStatisticsExtractors.weight(out);
      final double N = addedWeight + outWeight;
      if ( addedWeight  > 0 &&  addedWeight  < minBinSize)
        return -1000000;
      final double wOut = AdditiveStatisticsExtractors.weight(out);
      if (wOut > 0 && wOut < minBinSize)
        return -1000000;

      final double randomSplitVariance = (addedSum2+outSum2) - 2 * c * addedWeight * (addedSum + outSum) / N + addedWeight * c * c;

      final double splitVariance = (addedSum2+outSum2) - c * (2 * addedSum - addedWeight * c);
      final double score = (c * (2 * addedSum - addedWeight * c) - 2 * c * addedWeight * (addedSum + outSum) / N + addedWeight * c * c);

      return score > 0 ? score / (complexity + borders) : Double.NEGATIVE_INFINITY;

//      double addedScore = c * (2 * addedSum - addedWeight * c) / ( currentComplexity + borders);
//      return addedScore > 0 ? addedScore : Double.NEGATIVE_INFINITY;
    } else {
      AdditiveStatistics inside = subset.inside().append(added);
      final int borders = borders(feature, start, end);
      return score(inside, out, complexity + borders);
    }
  }

  private int borders(BFGrid.BFRow feature, int start, int end) {
    return start != 0 && end != feature.size() ? 16 : 1;
  }

  private double score(AdditiveStatistics inside, AdditiveStatistics outside, int complexity) {
    final double wIn = AdditiveStatisticsExtractors.weight(inside);
    if (wIn > 0 && wIn < minBinSize)
      return -1000000;
    final double wOut = AdditiveStatisticsExtractors.weight(outside);
    if (wOut > 0 && wOut < minBinSize)
      return -1000000;
    return -loss.score(inside) / complexity;
  }

  @Override
  public double score() {
    return score(subset.inside(), subset.outside(), complexity);
  }

  @Override
  public double insideIncrement() {
    return loss.bestIncrement(subset.inside());
  }

  @Override
  public void endClause() {
    complexity++;
    subset.endClause();
  }

  public void addCondition(BFGrid.BFRow feature, int start, int end) {
    subset().addCondition(feature, start, end);
    complexity += borders(feature, start, end);
    complexity ++;
  }

  @Override
  public CherryPointsHolder subset() {
    return subset;
  }
}

