package com.spbsu.ml.data;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.commons.util.BestHolder;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.models.CNF;
import gnu.trove.list.array.TIntArrayList;

import java.util.*;

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
  public class Result {
    public final int[] points;
    public final CNF.Clause clause;

    Result(int[] points, CNF.Clause clause) {
      this.points = points;
      this.clause = clause;
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

  public Result fit(final int[] points, final CherryLoss loss) {
    List<CherryBestHolder> features = new ArrayList<>(100);
    final CherrySubset subset = new CherrySubset(bds, factory, points);
    double currentScore = Double.NEGATIVE_INFINITY;
    while (true) {
      final CherryBestHolder bestHolder = new CherryBestHolder();
      subset.visitAll(new Aggregate.IntervalVisitor<AdditiveStatistics>() {
        @Override
        public void accept(BFGrid.BFRow feature, int start, int end, AdditiveStatistics added, AdditiveStatistics out) {
          if (!feature.empty()) {
            final double score = loss.score(feature, start, end, added, out);
            bestHolder.update(feature, score, start, end);
          }
        }
      });
      if (bestHolder.getScore() <= currentScore + 1e-9)
        break;
      AdditiveStatistics added = subset.add(bestHolder.getValue(), bestHolder.startBin(), bestHolder.endBin());
      features.add(bestHolder);
      loss.added(bestHolder.getValue(), bestHolder.startBin(), bestHolder.endBin(), added);
      currentScore = bestHolder.getScore();
    }
    return new Result(subset.pointsInside(), createClause(features));
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
      BitSet used = new BitSet(row.size() + 1);
      for (int k = i; k < j; ++k) {
        final int startBin = features.get(k).startBin();
        final int end = features.get(k).endBin() + 1;
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