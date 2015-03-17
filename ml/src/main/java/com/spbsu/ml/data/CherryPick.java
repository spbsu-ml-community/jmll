package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.BestHolder;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
import com.spbsu.ml.models.CNF;
import gnu.trove.list.array.TIntArrayList;

import java.util.*;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * User: solar
 * Date: 26.08.13
 * Time: 22:09
 */


class CherryBestHolder extends BestHolder<BFGrid.BFRow> {
  private int startBin;
  private int endBin;

  public synchronized boolean update(final BFGrid.BFRow update, final double score, final int startBin, final int endBin) {
    if (update(update, score)) {
      this.startBin = startBin;
      this.endBin = endBin;
      return true;
    }
    return false;
  }

  public int startBin() {
    return startBin;
  }
  public int endBin() {
    return endBin;
  }
}

public class CherryPick {
  public class ClauseComplexity {
    public final AdditiveStatistics inside;
    public final AdditiveStatistics outside;
    public final int complexity;
    public final double reg;

    ClauseComplexity(AdditiveStatistics outside, AdditiveStatistics inside, int complexity, double reg) {
      this.outside = outside;
      this.inside = inside;
      this.complexity = complexity;
      this.reg = reg;
    }
  }



  public class Result {
    public final int[] points;
    public final CNF.Clause clause;
    public final ClauseComplexity complexity;

    Result(int[] points, CNF.Clause clause, ClauseComplexity complexity) {
      this.points = points;
      this.clause = clause;
      this.complexity = complexity;
    }
  }

  private final BinarizedDataSet bds;
  private final BFGrid grid;
  private final Factory<AdditiveStatistics> factory;

  public CherryPick(final BinarizedDataSet bds, final Factory<AdditiveStatistics> factory) {
    this.bds = bds;
    this.grid = bds.grid();
    this.factory = factory;
  }

  public Result fit(final int[] points, final AdditiveStatistics priorOut,final double startReg, final StatBasedLoss<AdditiveStatistics> loss) {
    List<CherryBestHolder> features = new ArrayList<>(100);
    final CherrySubset subset = new CherrySubset(bds, factory, points);
    double currentScore = Double.NEGATIVE_INFINITY;
    final AdditiveStatistics total = subset.total();
    int complexity = 0;
    double currentReg = startReg;
    final double ideaSplit = Math.log(1.0 / 2);
    final double totalWeight = weight(total) + weight(priorOut);
    final double startWeight = weight(total);
    while (true) {
      final CherryBestHolder bestHolder = new CherryBestHolder();
      final double fCurrentReg = currentReg;
      final int finalComplexity = complexity;
      subset.visitAll(new Aggregate.IntervalVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out) {
          final AdditiveStatistics inside = subset.inside();
          inside.append(added);
          AdditiveStatistics outside = loss.statsFactory().create().append(out).append(priorOut);
          double score = loss.score(inside) + loss.score(outside);
//          double p = weight(added) / totalWeight;
//          double score;
//          {
//            double sumIn = sum(inside);
//            double sum2In = sum2(inside);
//            double sum2Out = sum2(out);
//            double wIn = weight(inside);
            double wAdded = weight(added);
            double wOut = weight(out);
            double p = wAdded /  (totalWeight);
          double p2 = wAdded / totalWeight;
            double borders = (start != 0 ? 1 : 0) + (end != feature.size() ? 1 : 0);
            double reg = fCurrentReg +   (borders-ideaSplit + p*Math.log(p)+ (1-p) * Math.log(1-p));
//            + p2*Math.log(p2)+ (1-p2) * Math.log(1-p2));
//            double var= wOut * ( sum2Out / (wOut - 1))
//                    + MathTools.sqr(wIn / (wIn - 1)) * (sum2In - sumIn * sumIn / wIn);
//            score = -var * (1 + 0.1*reg);
//          }
          score /= reg;
          bestHolder.update(feature, score, start,end);
        }
      });
      if (bestHolder.getScore() <= currentScore )
        break;
      features.add(bestHolder);
      AdditiveStatistics added = subset.add(bestHolder.getValue(), bestHolder.startBin(),bestHolder.endBin());
      double wAdded = weight(added);
      double wOut =  weight(subset.total());
      double p = (wAdded) / (totalWeight);
//      double p2 = (wAdded) / (totalWeight);
      double borders = (bestHolder.startBin() != 0 ? 1 : 0) + (bestHolder.endBin() != bestHolder.getValue().size() ? 1 : 0);
      currentReg +=  (borders- ideaSplit + p*Math.log(p) + (1-p)*Math.log(1-p));
//      currentReg += (2* Math.log( (wIn + wOut) / 2) - Math.log(wIn) - Math.log(wOut));
      currentScore = bestHolder.getScore();
      ++complexity;
    }
    return new Result(subset.pointsInside(), createClause(features), new ClauseComplexity(factory.create().append(total).remove(subset.inside()), subset.inside(), complexity, currentReg));
  }

  private CNF.Clause createClause(List<CherryBestHolder> features) {
    Collections.sort(features, new Comparator<CherryBestHolder>() {
      @Override
      public int compare(CherryBestHolder first, CherryBestHolder second) {
        int firstIndex = first.getValue().origFIndex;
        int secondIndex = second.getValue().origFIndex;

        if (firstIndex < secondIndex) {
          return -1;
        } else if (firstIndex > secondIndex) {
          return 1;
        } else {
          return Integer.compare(first.startBin(), second.startBin());
        }
      }
    });

    List<CNF.Condition> conditions = new ArrayList<>(features.size());
    for (int i = 0; i < features.size(); ++i) {
      int j = i + 1;
      BFGrid.BFRow row = features.get(i).getValue();
      int findex = row.origFIndex;
      while (j < features.size() && features.get(j).getValue().origFIndex == findex) {
        ++j;
      }
      BitSet used = new BitSet(row.size()+1);
      for (int k = i; k < j; ++k) {
        final int startBin = features.get(k).startBin();
        final int end = features.get(k).endBin()+1;
        used.set(startBin, end);
      }
      conditions.add(new CNF.Condition(row, used));
    }
    return new CNF.Clause(grid, conditions.toArray(new CNF.Condition[conditions.size()]));
  }
}


class CherrySubset {
  private final BinarizedDataSet bds;
  private final Factory<AdditiveStatistics> factory;
  private int[] points;
  private Aggregate outsideAggregate;
  private AdditiveStatistics inside;
  private TIntArrayList pointsInside = new TIntArrayList();

  public CherrySubset(BinarizedDataSet bds, Factory<AdditiveStatistics> factory, int[] points) {
    this.bds = bds;
    this.factory = factory;
    this.points = points;
    this.inside = factory.create();
    this.outsideAggregate = new Aggregate(bds, factory, points);
  }

  public AdditiveStatistics add(BFGrid.BFRow row, int startBin, int endBin) {
    final TIntArrayList outsidePoints = new TIntArrayList(points.length);
    int insideStart = pointsInside.size();
    final byte[] bins = bds.bins(row.origFIndex);
    AdditiveStatistics added = factory.create();
    for (final int i : points) {
      final int bin = bins[i];
      if (startBin <= bin && bin <= endBin) {
        added.append(i, 1);
        pointsInside.add(i);
      } else {
        outsidePoints.add(i);
      }
    }

    points = outsidePoints.toArray();
    if (pointsInside.size() - insideStart < points.length) {
      int[] in = new int[pointsInside.size() - insideStart];
      for (int i = insideStart; i < pointsInside.size(); ++i) {
        in[i - insideStart] = pointsInside.get(i);
      }
      final Aggregate removed = new Aggregate(bds, factory, in);
      outsideAggregate.remove(removed);
    } else {
      outsideAggregate = new Aggregate(bds, factory, points);
    }
    inside.append(added);
    return added;
  }

  public void visitAll(final Aggregate.IntervalVisitor<? extends AdditiveStatistics> visitor) {
    outsideAggregate.visit(visitor);
  }

  public AdditiveStatistics total() {
    return outsideAggregate.total();
  }

  public AdditiveStatistics inside() {
    AdditiveStatistics stat = factory.create().append(inside);
    return stat;
  }

  public int[] pointsInside() {
    return pointsInside.toArray();
  }
}