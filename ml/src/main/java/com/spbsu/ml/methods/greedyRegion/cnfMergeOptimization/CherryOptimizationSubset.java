package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.models.CNF;
import gnu.trove.list.array.TIntArrayList;

import java.util.BitSet;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.sum;
import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * Created by noxoomo on 30/11/14.
 */
@SuppressWarnings("UnusedDeclaration")
public class CherryOptimizationSubset {
  private final int power;
  final double initialCardinality;
  CNF.Clause clause;
  BinarizedDataSet bds;
  public int[] all;
  public int[] minimumIndices;
  public AdditiveStatistics stat;

  boolean isMinimumOutside;

  public CherryOptimizationSubset(final BinarizedDataSet bds, final Factory<AdditiveStatistics> statFactory, final CNF.Clause clause, final int[] points, final double cardinality) {
    initialCardinality = cardinality;
    final TIntArrayList inside = new TIntArrayList(points.length);
    final TIntArrayList outside = new TIntArrayList(points.length);
    stat = statFactory.create();
    for (int i = 0; i < points.length; i++) {
      if (clause.contains(bds, points[i])) {
        stat.append(points[i], 1);
        inside.add(points[i]);
      } else outside.add(points[i]);
    }
    this.all = points;
    this.bds = bds;
    this.clause = clause;
    isMinimumOutside = outside.size() < inside.size();
    minimumIndices = isMinimumOutside ? outside.toArray() : inside.toArray();
//    this.cardinality = clause.cardinality();
    this.power = inside.size();
  }

  CherryOptimizationSubset(final BinarizedDataSet bds, final CNF.Clause clause, final int[] minimumIndices, final boolean isMinimumOutside, final int[] all, final AdditiveStatistics stat, final double initialCardinality) {
    this.all = all;
    this.bds = bds;
    this.stat = stat;
    this.clause = clause;
    this.initialCardinality = initialCardinality;
    if (minimumIndices.length > all.length / 2) {
      this.minimumIndices = new int[all.length - minimumIndices.length];
      int oldIndex = 0;
      int newIndex = 0;
      for (int i = 0; i < all.length; i++) {
        if (oldIndex < minimumIndices.length && all[i] == minimumIndices[oldIndex])
          oldIndex++;
        else
          this.minimumIndices[newIndex++] = all[i];
      }
      this.isMinimumOutside = !isMinimumOutside;
    } else {
      this.minimumIndices = minimumIndices;
      this.isMinimumOutside = isMinimumOutside;
    }

//    this.cardinality = clause.cardinality();
    this.power = isMinimumOutside ? all.length - minimumIndices.length : minimumIndices.length;
  }

  public int[] inside() {
    if (!isMinimumOutside)
      return minimumIndices;
    final int[] inside = new int[all.length - minimumIndices.length];
    int minIndex = 0;
    int inIndex = 0;

    for (int i = 0; i < all.length; i++) {
      if (minIndex < minimumIndices.length && all[i] == minimumIndices[minIndex])
        minIndex++;
      else
        inside[inIndex++] = all[i];
    }
    return inside;
  }

  public int[] outside() {
    if (isMinimumOutside)
      return minimumIndices;
    final int[] outside = new int[all.length - minimumIndices.length];
    int minIndex = 0;
    int outIndex = 0;

    for (int i = 0; i < all.length; i++) {
      if (minIndex < minimumIndices.length && all[i] == minimumIndices[minIndex])
        minIndex++;
      else
        outside[outIndex++] = all[i];
    }
    return outside;
  }

  public double cardinality() {
    return initialCardinality + clause.cardinality();
  }

  public int power() {
    return (int) ((L2.MSEStats) ((WeightedLoss.Stat) stat).inside).weight;
  }

  @Override
  public String toString() {
    return clause.toString() + ": (power:" + power + ")";
  }

  public boolean nextTo(final CherryOptimizationSubset current) {
    if (clause.conditions.length != 1 || current.clause.conditions.length != 1)
      return false;
    if (clause.conditions[0].findex != current.clause.conditions[0].findex)
      return false;
    final BitSet mask = clause.conditions[0].used;
    final BitSet otherMask = current.clause.conditions[0].used;
    for (int i = mask.nextSetBit(0); i >= 0; i = mask.nextSetBit(i + 1)) {
      if (i > 0 && otherMask.get(i + 1) || i < otherMask.size() - 1 && otherMask.get(i + 1))
        return true;
    }
    return false;
  }

  private static volatile int counter = 0;
  private final int index = counter++;

  public int index() {
    return index;
  }

  public void checkIntegrity() {
    for (int i = 0, j = 0; i < all.length; i++) {
      final boolean value = clause.contains(bds, all[i]);
      if (j < minimumIndices.length && minimumIndices[j] == all[i]) {
        clause.contains(bds, all[i]);
        if (value && isMinimumOutside)
          System.out.println();
        j++;
      } else {
        if (!value && isMinimumOutside)
          System.out.println();
      }
    }
  }

  public void checkStat(Factory<AdditiveStatistics> factory) {
    AdditiveStatistics inside = factory.create();
    int[] pointsInside = inside();
    for (int i : pointsInside)
      inside.append(i, 1);
    assert (Math.abs(sum(inside) - sum(stat)) < 1e-9);
    assert (weight(inside) == weight(stat));
  }


}
