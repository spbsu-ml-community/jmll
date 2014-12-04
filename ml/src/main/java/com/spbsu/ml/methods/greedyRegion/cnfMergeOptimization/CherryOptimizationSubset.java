package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.models.CNF;

import static com.spbsu.ml.methods.greedyRegion.AdditiveStatisticsExtractors.weight;

/**
 * Created by noxoomo on 30/11/14.
 */
public class CherryOptimizationSubset {
  CNF.Clause clause;
  BinarizedDataSet bds;
  public int[] all;
  public int[] minimumIndices;
  boolean isMinimumOutside;

  public CherryOptimizationSubset(BinarizedDataSet bds, AdditiveStatistics stat, CNF.Clause clause, int[] points) {
    int inside = 0;
    for (int i = 0; i < points.length; i++) {
      if (clause.value(bds, points[i]) == 1.0) {
        stat.append(points[i], 1);
        inside++;
      }
    }
    if (inside > points.length / 2) {
      isMinimumOutside = true;
      minimumIndices = new int[points.length - inside];
      int index = 0;
      for (int i = 0; i < points.length; i++) {
        if (clause.value(bds, points[i]) != 1.0)
          minimumIndices[index++] = points[i];
      }
    } else {
      isMinimumOutside = false;
      minimumIndices = new int[inside];
      int index = 0;
      for (int i = 0; i < points.length; i++) {
        if (clause.value(bds, points[i]) == 1.0)
          minimumIndices[index++] = points[i];
      }
    }
    this.all = points;
    this.stat = stat;
    this.bds = bds;
    this.clause = clause;
    {
      double weight = weight(stat);
      this.cardinality = ModelComplexityCalcer.cardinality(bds.grid(), clause);
      this.regularization = -Math.log(weight + 1);
    }
  }

  CherryOptimizationSubset(BinarizedDataSet bds, CNF.Clause clause, int[] minimumIndices, boolean isMinimumOutside, int[] all, AdditiveStatistics stat, double reg) {
    this.all = all;
    this.bds = bds;
    this.stat = stat;
    this.clause = clause;
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

    for (int i = 0, j = 0; i < all.length; i++) {
      final boolean value = clause.value(bds, all[i]) == 1.;
      if (j < minimumIndices.length && minimumIndices[j] == all[i]) {
        if (value && isMinimumOutside)
          System.out.println();
        j++;
      }
      else {
        if (!value && isMinimumOutside)
          System.out.println();
      }
    }


    this.cardinality = ModelComplexityCalcer.cardinality(bds.grid(), clause);
    this.regularization = reg;
  }

  CherryOptimizationSubset(BinarizedDataSet bds, CNF.Clause clause, int[] minimumIndices, boolean isMinimumOutside, int[] all, AdditiveStatistics stat) {
    this.all = all;
    this.bds = bds;
    this.stat = stat;
    this.clause = clause;
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

//    for (int i = 0, j = 0; i < all.length; i++) {
//      final boolean value = clause.value(bds, all[i]) == 1.;
//      if (j < minimumIndices.length && minimumIndices[j] == all[i]) {
//        if (value && isMinimumOutside)
//          System.out.println();
//        j++;
//      }
//      else {
//        if (!value && isMinimumOutside)
//          System.out.println();
//      }
//    }


    this.cardinality = ModelComplexityCalcer.cardinality(bds.grid(), clause);
    double weight = weight(stat);
    this.regularization = -Math.log(weight+1);
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

  public AdditiveStatistics stat;
  private double regularization;
  private double cardinality;

  public double regularization() {
    return regularization;
  }

  public double cardinality() {
    return cardinality;
  }


}
