package com.spbsu.ml.methods.greedyRegion.cherry;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.cherry.CherryLoss;
import com.spbsu.ml.data.cherry.CherryPointsHolder;
import com.spbsu.ml.loss.StatBasedLoss;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

public class OutLoss<Subset extends CherryPointsHolder, Loss extends StatBasedLoss<AdditiveStatistics>> implements CherryLoss {
  private Subset subset;
  private Loss loss;
  private int complexity = 1;
  private int minBinSize = 50;

  OutLoss(Subset subset, Loss loss) {
    this.subset = subset;
    this.loss = loss;
  }

  @Override
  public double score(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out) {
    if (start == 0 && end == feature.size())
      return Double.NEGATIVE_INFINITY;
    AdditiveStatistics inside = subset.inside().append(added);
    final int borders = borders(feature, start, end);
    return score(inside, out, complexity + borders);
  }

  private int borders(BFGrid.BFRow feature, int start, int end) {
    return start != 0 && end != feature.size() ? 4 : 1;
  }

  private double score(AdditiveStatistics inside, AdditiveStatistics outside, int complexity) {
    final double wIn = weight(inside);
    if (wIn > 0 && wIn < minBinSize)
      return Double.NEGATIVE_INFINITY;
    final double wOut = weight(inside);
    if (wIn > 0 && wIn < minBinSize)
      return Double.NEGATIVE_INFINITY;
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
    ++complexity;
    subset.endClause();
  }

  public void addCondition(BFGrid.BFRow feature, int start, int end) {
    subset().addCondition(feature, start, end);
    complexity += borders(feature, start, end);
  }

  @Override
  public CherryPointsHolder subset() {
    return subset;
  }
}

