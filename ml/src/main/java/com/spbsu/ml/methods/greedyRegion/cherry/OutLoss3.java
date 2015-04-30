package com.spbsu.ml.methods.greedyRegion.cherry;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.cherry.CherryLoss;
import com.spbsu.ml.data.cherry.CherryPointsHolder;
import com.spbsu.ml.loss.StatBasedLoss;
import gnu.trove.set.hash.TIntHashSet;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;
import static com.spbsu.ml.methods.greedyRegion.GreedyTDWeakRegionMTA.sum;

public class OutLoss3<Subset extends CherryPointsHolder, Loss extends StatBasedLoss<AdditiveStatistics>> extends CherryLoss {
  private Subset subset;
  private Loss loss;
  private int complexity = 1;
  private int minBinSize = 50;
  private TIntHashSet used = new TIntHashSet();

  OutLoss3(Subset subset, Loss loss) {
    this.subset = subset;
    this.loss = loss;
  }

  @Override
  public double score(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out) {
    if (start == 0 && end == feature.size())
      return Double.NEGATIVE_INFINITY;
    int newsize = used.contains(feature.origFIndex) ? used.size() : used.size()+1;
    if (newsize > 7)
      return Double.NEGATIVE_INFINITY;

    AdditiveStatistics inside = subset.inside();
    AdditiveStatistics total = subset.inside().append(added);
    final double R1 = -sum(total) * sum(total) / weight(total);
    total.append(out);
    final double R2 = Math.min(weight(inside) > 1 ? -sum(inside) * sum(inside) / weight(inside) : 0,-sum(total) * sum(total) / weight(total)) ;
    final int borders = borders(feature, start, end);
    final double score =  (R2-R1) / (borders);
    return score >= 0 ? score :  -1000000;//score(total, out, complexity + borders);
  }

  private int borders(BFGrid.BFRow feature, int start, int end) {
    return start != 0 && end != feature.size() ? 4 : 1;
  }

  private double score(AdditiveStatistics inside, AdditiveStatistics outside, int complexity) {
    final double wIn = weight(inside);
    if (used.size() > 6)
      return Double.NEGATIVE_INFINITY;
    if (wIn > 0 && wIn < minBinSize)
      return -1000000;
    final double wOut = weight(outside);
    if (wOut > 0 && wOut < minBinSize)
      return -1000000;
    double s = sum(inside) + sum(outside);
    double w = wIn + wOut;
    final double score = weight(inside) > 0 ? sum(inside) * sum(inside) / weight(inside) : 0;
    return score;
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
    subset.endClause();
    complexity ++;
  }

  public void addCondition(BFGrid.BFRow feature, int start, int end) {
    subset().addCondition(feature, start, end);
    complexity += borders(feature, start, end);
    used.add(feature.origFIndex);
    complexity ++;
  }

  @Override
  public CherryPointsHolder subset() {
    return subset;
  }
}

