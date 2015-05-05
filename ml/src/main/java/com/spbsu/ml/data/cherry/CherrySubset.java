package com.spbsu.ml.data.cherry;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.data.Aggregate;
import com.spbsu.ml.data.impl.BinarizedDataSet;

public class CherrySubset implements CherryPointsHolder {
  private final BinarizedDataSet bds;
  private final Factory<AdditiveStatistics> factory;
  private int[] points;
  private int[] cache;
  private boolean insideSign = false;
  private int outsideStart = 0;
  private Aggregate outsideAggregate;
  private AdditiveStatistics inside;
  private int length;

  public CherrySubset(BinarizedDataSet bds, Factory<AdditiveStatistics> factory, int[] points) {
    this.bds = bds;
    this.factory = factory;
    this.points = points;
    for (int i = 0; i < points.length; ++i)
      points[i]++;
    cache = new int[points.length];
    this.length = points.length;
  }


  public void visitAll(final Aggregate.IntervalVisitor<? extends AdditiveStatistics> visitor) {
    outsideAggregate.visit(visitor);
  }

  @Override
  public BFGrid grid() {
    return bds.grid();
  }

  @Override
  public void endClause() {
    insideSign = !insideSign;
    length = outsideStart;
    outsideStart = 0;
  }

  @Override
  public void startClause() {
    inside = factory.create();
    this.outsideAggregate = new Aggregate(bds, factory, getPoints(0, length));
  }

  @Override
  public AdditiveStatistics addCondition(final BFGrid.BFRow row,
                                         final int startBin, final int endBin) {
    final byte[] bins = bds.bins(row.origFIndex);
    AdditiveStatistics added = factory.create();
    int count = 0;
    for (int i = outsideStart; i < length; ++i) {
      final int point = getPoint(i);
      final int bin = bins[point];
      if (startBin <= bin && bin <= endBin) {
        added.append(point, 1);
        points[i] *= -1;
        ++count;
      }
    }
    gather(outsideStart + count);
    if (count > length - outsideStart) {
      int[] out = getPoints(outsideStart + count, length);
      outsideAggregate = new Aggregate(bds, factory, out);
    } else {
      int[] addedPoints = getPoints(outsideStart, count);
      final Aggregate toRemove = new Aggregate(bds, factory, addedPoints);
      outsideAggregate.remove(toRemove);
    }
    outsideStart += count;
    inside.append(added);
    return inside;
  }

  private int[] getPoints(final int from, final int size) {
    final int[] result = new int[size];
    int len = (result.length / 4) * 4;
    for (int i = 0; i < len; i += 4) {
      result[i] = getPoint(from + i);
      result[i + 1] = getPoint(from + i + 1);
      result[i + 2] = getPoint(from + i + 2);
      result[i + 3] = getPoint(from + i + 3);
    }
    for (int i = len; i < result.length; ++i) {
      result[i] = getPoint(from + i);
    }
    return result;
  }


  private int getPoint(int i) {
    final int idx = points[i] > 0 ? points[i] : -points[i];
    return idx - 1;
  }

  private void gather(final int inCount) {
    int inPtr = 0;
    int outPtr = inCount;
    for (int i = 0; i < length; ++i) {
      if ((points[i] > 0) == insideSign) {
        cache[inPtr++] = points[i];
      } else {
        cache[outPtr++] = points[i];
      }
    }
    final int[] tmp = points;
    points = cache;
    cache = tmp;
  }

  public AdditiveStatistics total() {
    return outsideAggregate.total().append(inside);
  }

  public AdditiveStatistics inside() {
    final AdditiveStatistics stat = factory.create().append(inside);
    return stat;
  }

  public AdditiveStatistics outside() {
    final AdditiveStatistics stat = outsideAggregate.total();
    return stat;
  }
}
