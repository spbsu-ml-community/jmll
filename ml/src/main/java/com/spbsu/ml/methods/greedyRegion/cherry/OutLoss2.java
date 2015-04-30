package com.spbsu.ml.methods.greedyRegion.cherry;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.cherry.CherryLoss;
import com.spbsu.ml.data.cherry.CherryPointsHolder;
import com.spbsu.ml.loss.StatBasedLoss;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.*;

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
    if (weight(currentInside) != 0) {

      final int borders = borders(feature, start, end);
      final double c =  insideIncrement();
      final double addedWeight = weight(added);
      final double addedSum = sum(added);
      final double addedSum2 = sum2(added);
      final double outSum = sum(out);
      final double outSum2 = sum2(out);
      final double outWeight = weight(out);
      final double N = addedWeight + outWeight;
      if ( addedWeight  > 0 &&  addedWeight  < minBinSize)
        return -1000000;
      final double wOut = weight(out);
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
    final double wIn = weight(inside);
    if (wIn > 0 && wIn < minBinSize)
      return -1000000;
    final double wOut = weight(outside);
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

