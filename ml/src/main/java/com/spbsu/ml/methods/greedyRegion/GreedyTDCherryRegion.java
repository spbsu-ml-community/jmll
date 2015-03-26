package com.spbsu.ml.methods.greedyRegion;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Binarize;
import com.spbsu.ml.data.CherryPick;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.CNF;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.commons.math.MathTools.sqr;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * User: solar
 * Date: 15.11.12
 * Time: 15:19
 */
public class GreedyTDCherryRegion<Loss extends StatBasedLoss> extends VecOptimization.Stub<Loss> {
  public final BFGrid grid;

  public GreedyTDCherryRegion(final BFGrid grid) {
    this.grid = grid;
  }

  private int[] learnPoints(Loss loss, VecDataSet ds) {
    if (loss instanceof WeightedLoss) {
      return ((WeightedLoss) loss).points();
    } else return ArrayTools.sequence(0, ds.length());
  }

  @Override
  public CNF fit(final VecDataSet learn, final Loss loss) {
    final List<CNF.Clause> conditions = new ArrayList<>(100);
    final BinarizedDataSet bds = learn.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);
    final CherryPick pick = new CherryPick(bds, loss.statsFactory());
    int[] points = learnPoints(loss, learn);
    double currentScore = Double.NEGATIVE_INFINITY;
    CherryTDLoss localLoss;
    {
      AdditiveStatistics total = (AdditiveStatistics) loss.statsFactory().create();
      for (int point: points)
        total.append(point, 1);
//      localLoss = new InOutLoss(total, loss);
      localLoss = new OutLoss(total, loss);
//      localLoss = new HaarLoss(total, loss.statsFactory());
    }
    double bestIncInside = 0;
    double bestIncOutside = 0;
    while (true) {
      final CherryPick.Result result = pick.fit(points, localLoss);
      if (localLoss.score() <= currentScore + 1e-9) {
        break;
      }
      System.out.println("\nAdded clause " + result.clause);
      points = result.points;
      conditions.add(result.clause);
      currentScore = localLoss.score();
      bestIncInside = localLoss.insideIncrement();
      bestIncOutside = localLoss.outsideIncrement();
      localLoss.nextIteration();
    }
    return new CNF(conditions.toArray(new CNF.Clause[conditions.size()]), bestIncInside, bestIncOutside, grid);
  }
}

class HaarLoss implements CherryTDLoss {
  private AdditiveStatistics inside;
  private final AdditiveStatistics total;
  private Factory<AdditiveStatistics> factory;
  private int complexity = 1;
  private int orCount = 0;

  public HaarLoss(AdditiveStatistics total, Factory<AdditiveStatistics> factory) {
    this.total = total;
    this.factory = factory;
    inside = factory.create();
  }

  @Override
  public double score(final BFGrid.BFRow feature, final int start, final int end,
                      final AdditiveStatistics added, final AdditiveStatistics out) {
    if (orCount > 1 || weight(added) == 0)
      return Double.NEGATIVE_INFINITY;
    final AdditiveStatistics in = factory.create().append(inside).append(added);
    final AdditiveStatistics outside = factory.create().append(total).remove(in);
    int borders = ((start != 0 ? 1 : 0) + (end != feature.size() ? 1 : 0));
    return score(in, outside, complexity + borders);
  }

  @Override
  public void added(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added) {
    complexity += (start != 0 ? 1 : 0) + (end != feature.size() ? 1 : 0);
    inside.append(added);
    ++orCount;
  }

  @Override
  public void nextIteration() {
    this.inside = factory.create();
    orCount = 0;
    ++iteration;
  }

  int iteration = 0;

  //we are maximizing score
  private double score(final AdditiveStatistics in, final AdditiveStatistics out, final double complexity) {
    if (complexity > 10)
      return Double.NEGATIVE_INFINITY;
    final double sumIn = sum(in);
    final double sumOut = sum(out);
    final double wIn = weight(in);
    final double wOut = weight(out);
    final double p = wIn / (wIn + wOut);
    if (wIn < 3 || wOut < 3)
      return Double.NEGATIVE_INFINITY;
    return sqr(sumIn - sumOut) / (1 + complexity);
  }

  @Override
  public double insideIncrement() {
    AdditiveStatistics outside = factory.create().append(total).remove(inside);
    final double inc = bestInc(inside, outside);
    System.out.println("Increment: " + inc);
    System.out.println("Inside weight: " + weight(inside));
    System.out.println("Outside weight: " + weight(outside));
    return inc;
  }

  @Override
  public double outsideIncrement() {
    AdditiveStatistics outside = factory.create().append(total).remove(inside);
    return -bestInc(inside, outside);
  }

  @Override
  public double score() {
    AdditiveStatistics outside = factory.create().append(total).remove(inside);
    return score(inside, outside, complexity);
  }

  private double bestInc(final AdditiveStatistics in, final AdditiveStatistics out) {
    final double total = weight(in) + weight(out);
    final double diff = sum(in) - sum(out);
    return diff / total;
  }
}


class InOutLoss implements CherryTDLoss {
  private AdditiveStatistics inside;
  private final AdditiveStatistics total;
  private Factory<AdditiveStatistics> factory;
  private int complexity = 1;
  private double infoLoss = 0;
  private StatBasedLoss<AdditiveStatistics> base;

  public InOutLoss(AdditiveStatistics total, StatBasedLoss<AdditiveStatistics> base) {
    this.total = total;
    this.factory = base.statsFactory();
    inside = factory.create();
    this.base = base;
  }

  @Override
  public double score(final BFGrid.BFRow feature, final int start, final int end,
                      final AdditiveStatistics added, final AdditiveStatistics out) {
    if (orCount > 1)
      return Double.NEGATIVE_INFINITY;

    final AdditiveStatistics in = factory.create().append(inside).append(added);
    final AdditiveStatistics outside = factory.create().append(total).remove(in);
    double wadded = weight(added);
    double wtotal = weight(total);
    final double p = wadded / wtotal;
    double inf = -Math.log(0.5) + p * Math.log(p) + (1 - p) * Math.log(1 - p);
    return score(in, outside) / (complexity + infoLoss + inf + (start != 0 ? 1 : 0) + (end != feature.size() ? 1 : 0));
  }

  @Override
  public void added(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added) {
    complexity += (start != 0 ? 1 : 0) + (end != feature.size() ? 1 : 0);
    double win = weight(added);
    double wtotal = weight(total);
    final double p = win / wtotal;
    infoLoss += -Math.log(0.5) + p * Math.log(p) + (1 - p) * Math.log(1 - p);
    inside.append(added);
    ++orCount;
  }

  int orCount = 0;

  @Override
  public void nextIteration() {
    this.inside = factory.create();
    orCount = 0;
  }

  //we are maximizing score
  private double score(final AdditiveStatistics in, final AdditiveStatistics out) {
    return -(base.score(in) + base.score(out));
  }

  @Override
  public double insideIncrement() {
    return base.bestIncrement(inside);
  }

  @Override
  public double outsideIncrement() {
    AdditiveStatistics outside = factory.create().append(total).remove(inside);
    return base.bestIncrement(outside);
  }

  @Override
  public double score() {
    AdditiveStatistics outside = factory.create().append(total).remove(inside);
    return score(inside, outside) / (complexity + infoLoss);
  }


}


class OutLoss implements CherryTDLoss {
  private AdditiveStatistics inside;
  private final AdditiveStatistics total;
  private Factory<AdditiveStatistics> factory;
  private int complexity = 1;
  private double infoLoss = 0;
  private StatBasedLoss<AdditiveStatistics> base;

  public OutLoss(AdditiveStatistics total, StatBasedLoss<AdditiveStatistics> base) {
    this.total = total;
    this.factory = base.statsFactory();
    inside = factory.create();
    this.base = base;
  }

  @Override
  public double score(final BFGrid.BFRow feature, final int start, final int end,
                      final AdditiveStatistics added, final AdditiveStatistics out) {
    if (orCount > 3)
      return Double.NEGATIVE_INFINITY;
    final AdditiveStatistics in = factory.create().append(inside).append(added);
    final AdditiveStatistics outside = factory.create().append(total).remove(in);
    double wadded = weight(added);
    if (wadded < 3)
      return Double.NEGATIVE_INFINITY;
    final int borders = 1+ (start != 0 ? 1 : 0) + (end != feature.size() ? 1 : 0);
    if (complexity + borders > 10) {
      return Double.NEGATIVE_INFINITY;
    }
    if (weight(inside) == 0) {
      return score(in, outside, complexity + borders);
    } else {
      double sumIn = sum(inside);
      double weightIn = weight(inside);

      double sumAdded = sum(added);
      double weightAdded = weight(added);

      double wtotal = weight(total);
      final double p = wadded / wtotal;
      double inf = - p * Math.log(p) - (1 - p) * Math.log(1 - p);

      double score = sumIn / weightIn * ((1 - weightAdded / weightIn) * sumIn + 2 * sumAdded);
      return  score * (1 +  2*Math.log(wadded))/ ( borders + complexity);
    }

//    return score(in, outside, complexity + borders) * (1 + 2 * Math.log(wadded)) ;// / (complexity + infoLoss + inf +  ) ;
  }

  @Override
  public void added(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added) {
    complexity +=  1+ (start != 0 ? 1 : 0) + (end != feature.size() ? 1 : 0);
    double win = weight(added);
    double wtotal = weight(total);
    final double p = win / wtotal;
    infoLoss +=- p * Math.log(p)  - (1 - p) * Math.log(1 - p);
    inside.append(added);
    System.out.print("\nAdded " + weight(added) + " points");
    ++orCount;
  }

  @Override
  public void nextIteration() {
    this.inside = factory.create();
    this.orCount = 0;
  }

  int orCount = 0;

  //we are maximizing score
  private double score(final AdditiveStatistics in, final AdditiveStatistics out, int complexity) {
    if (complexity > 10) {
      return Double.NEGATIVE_INFINITY;
    }
    return -(base.score(in)) * (1 +2* Math.log(weight(in)) )/complexity;
   }

  @Override
  public double insideIncrement() {
    return base.bestIncrement(inside);
  }

  @Override
  public double outsideIncrement() {
    return 0;
  }

  @Override
  public double score() {
    if (complexity > 10) {
      return Double.NEGATIVE_INFINITY;
    }
    return weight(inside) > 0 ? sqr(sum(inside)) * (1 + 2 * Math.log(weight(inside))) / weight(inside) / (complexity) : Double.NEGATIVE_INFINITY;
//    AdditiveStatistics outside = factory.create().append(total).remove(inside);
//    return score(inside, outside,complexity);// /  (complexity + infoLoss );
  }


}
