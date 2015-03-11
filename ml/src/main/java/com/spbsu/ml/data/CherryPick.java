package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.BestHolder;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.models.CNF;
import gnu.trove.list.array.TIntArrayList;

import java.util.*;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * User: solar
 * Date: 26.08.13
 * Time: 22:09
 */


class CherryBestHolder extends BestHolder<BFGrid.BinaryFeature> {
  private int length;

  public synchronized boolean update(final BFGrid.BinaryFeature update, final double score, final int length) {
    if (update(update, score)) {
      this.length = length;
      return true;
    }
    return false;
  }

  public int getLength() {
    return length;
  }
}

public class CherryPick {
  public class ClauseComplexity {
    public final AdditiveStatistics total;
    public final AdditiveStatistics inside;
    public final int complexity;
    public final double logsSum;

    ClauseComplexity(AdditiveStatistics total, AdditiveStatistics inside, int complexity, double logsSums) {
      this.total = total;
      this.inside = inside;
      this.complexity = complexity;
      this.logsSum = logsSums;
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

  public Result fit(final int[] points, final Evaluator<AdditiveStatistics> eval) {
    List<Pair<BFGrid.BinaryFeature, Integer>> features = new ArrayList<>(100);
    final CherrySubset subset = new CherrySubset(bds, factory, points);
    double currentScore = Double.NEGATIVE_INFINITY;
    final AdditiveStatistics total = subset.total();
    int complexity = 0;
    double currentReg = 0;
    final double totalWeight = weight(total);
    while (true) {
      final CherryBestHolder bestHolder = new CherryBestHolder();
      final double fCurrentReg = currentReg;
      final int finalComplexity = complexity;
      subset.visitAll(new Aggregate.IntervalVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BinaryFeature bf, int length, AdditiveStatistics added, AdditiveStatistics out) {
          final AdditiveStatistics inside = subset.inside();
          inside.append(added);
          double reg = Math.log(weight(inside)) + Math.log(weight(added));// + fCurrentReg + Math.log(weight(out));
          reg /= (finalComplexity+2);
          final double score = eval.value(inside) * (1 + 32*reg);
          bestHolder.update(bf, score, length);
        }
      });
      if (bestHolder.getScore() <= currentScore)
        break;
      BFGrid.BinaryFeature feature = bestHolder.getValue();
      features.add(new Pair<>(feature, bestHolder.getLength()));
      AdditiveStatistics added = subset.add(feature, bestHolder.getLength());
      currentReg += Math.log(weight(added));
      currentScore = bestHolder.getScore();
      ++complexity;
    }
    double outWeight = weight(factory.create().append(total).remove(subset.inside()));
    return new Result(subset.pointsInside(), createClause(features), new ClauseComplexity(total, subset.inside(), complexity, currentReg + Math.log(outWeight)));
  }

  private CNF.Clause createClause(List<Pair<BFGrid.BinaryFeature, Integer>> features) {
    Collections.sort(features, new Comparator<Pair<BFGrid.BinaryFeature, Integer>>() {
      @Override
      public int compare(Pair<BFGrid.BinaryFeature, Integer> first, Pair<BFGrid.BinaryFeature, Integer> second) {
        int firstIndex = first.getFirst().findex;
        int secondIndex = second.getFirst().findex;

        if (firstIndex < secondIndex) {
          return -1;
        } else if (firstIndex > secondIndex) {
          return 1;
        } else {
          return Integer.compare(first.getSecond(), second.getSecond());
        }
      }
    });

    List<CNF.Condition> conditions = new ArrayList<>(features.size());
    for (int i = 0; i < features.size(); ++i) {
      int j = i + 1;
      BFGrid.BFRow row = features.get(i).first.row();
      int findex = row.origFIndex;
      while (j < features.size() && features.get(j).first.findex == findex) {
        ++j;
      }
      BitSet used = new BitSet(row.size());
      for (int k = i; k < j; ++k) {
        final int offset = features.get(k).getFirst().binNo;
        final int len = features.get(k).getSecond();
        used.set(offset, offset + len);
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

  public AdditiveStatistics add(BFGrid.BinaryFeature bf, int len) {
    final TIntArrayList outsidePoints = new TIntArrayList(points.length);
    int insideStart = pointsInside.size();
    final byte[] bins = bds.bins(bf.findex);
    AdditiveStatistics added = factory.create();
    for (final int i : points) {
      final int bin = bins[i];
      if (!(bf.binNo <= bin && bin < bf.binNo + len)) {
        outsidePoints.add(i);
      } else {
        added.append(i, 1);
        pointsInside.add(i);
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