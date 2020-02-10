package com.expleague.ml.methods.greedyRegion.cnfMergeOptimization;

import com.expleague.commons.func.AdditiveStatistics;
import com.expleague.commons.func.Factory;
import com.expleague.ml.methods.greedyMergeOptimization.MergeOptimization;
import com.expleague.ml.models.CNF;
import com.expleague.commons.util.ArrayTools;
import gnu.trove.list.array.TIntArrayList;

import java.util.function.IntFunction;

/**
 * Created by noxoomo on 30/11/14.
 */
public class CherryOptimizationSubsetMerger implements MergeOptimization<CherryOptimizationSubset> {
  private final IntFunction<AdditiveStatistics> factory;

  public CherryOptimizationSubsetMerger(final IntFunction<AdditiveStatistics> factory) {
    this.factory =  factory;
  }

  @Override
  public CherryOptimizationSubset merge(CherryOptimizationSubset first, CherryOptimizationSubset second) {
    final CNF.Clause clause = new CNF.Clause(first.bds.grid(), ArrayTools.concat(first.clause.conditions, second.clause.conditions));

    final AdditiveStatistics stat = factory.apply(0); // TODO: remove this trash
    if (first.minimumIndices.length < second.minimumIndices.length) {
      final CherryOptimizationSubset tmp = first;
      first = second;
      second = tmp;
    }
    if (first.isMinimumOutside && second.isMinimumOutside) { // intersection of outer
      stat.append(second.stat);
      final int[] firstOutside = first.minimumIndices;
      final int[] secondOutside = second.minimumIndices;
      final TIntArrayList mergedOutside = new TIntArrayList(second.minimumIndices.length);
      int firstIndex = 0;
      for (int i = 0; i <= secondOutside.length; i++) {
        final boolean last = i == secondOutside.length;
        final int next = !last ? secondOutside[i] : Integer.MAX_VALUE;
        while (firstIndex < firstOutside.length && firstOutside[firstIndex] < next) {
          firstIndex++;
        }
        if (!last) {
          if (firstIndex < firstOutside.length && firstOutside[firstIndex] == next)
            mergedOutside.add(next);
          else stat.append(next, 1);
        }
      }
      return new CherryOptimizationSubset(first.bds, clause, mergedOutside.toArray(), true, first.all, stat, first.initialCardinality);
    }
    else if (first.isMinimumOutside && !second.isMinimumOutside) {
      stat.append(first.stat);
      final int[] firstOutside = first.minimumIndices;
      final int[] secondInside = second.minimumIndices;
      final TIntArrayList mergedOutside = new TIntArrayList(first.minimumIndices.length);
      int firstIndex = 0;
      for (int i = 0; i <= secondInside.length; i++) {
        final boolean last = i == secondInside.length;
        final int next = !last ? secondInside[i] : Integer.MAX_VALUE;
        while (firstIndex < firstOutside.length && firstOutside[firstIndex] < next) {
          mergedOutside.add(firstOutside[firstIndex]);
          firstIndex++;
        }
        if (!last) {
          if (firstIndex < firstOutside.length && firstOutside[firstIndex] == next) {
            stat.append(next, 1);
            firstIndex++;
          }
        }
      }
      return new CherryOptimizationSubset(first.bds, clause, mergedOutside.toArray(), true, first.all, stat, first.initialCardinality);
    }
    else if (!first.isMinimumOutside && second.isMinimumOutside) {
      stat.append(second.stat);
      final int[] firstInside = first.minimumIndices;
      final int[] secondOutside = second.minimumIndices;
      final TIntArrayList mergedOutside = new TIntArrayList(second.minimumIndices.length);
      int secondIndex = 0;
      for (int i = 0; i <= firstInside.length; i++) {
        final boolean last = i == firstInside.length;
        final int next = !last ? firstInside[i] : Integer.MAX_VALUE;
        while (secondIndex < secondOutside.length && secondOutside[secondIndex] < next) {
          mergedOutside.add(secondOutside[secondIndex]);
          secondIndex++;
        }
        if (!last) {
          if (secondIndex < secondOutside.length && secondOutside[secondIndex] == next) {
            stat.append(next, 1);
            secondIndex++;
          }
        }
      }
      return new CherryOptimizationSubset(first.bds, clause, mergedOutside.toArray(), true, first.all, stat, first.initialCardinality);
    }
    else if (!first.isMinimumOutside && !second.isMinimumOutside) {
      stat.append(first.stat);
      final int[] firstOutside = first.minimumIndices;
      final int[] secondOutside = second.minimumIndices;
      final TIntArrayList mergedInside = new TIntArrayList(second.minimumIndices.length);
      int firstIndex = 0;
      for (int i = 0; i <= secondOutside.length; i++) {
        final boolean last = i == secondOutside.length;
        final int next = !last ? secondOutside[i] : Integer.MAX_VALUE;
        while (firstIndex < firstOutside.length && firstOutside[firstIndex] < next) {
          firstIndex++;
        }
        if (!last) {
          if (firstIndex < firstOutside.length && firstOutside[firstIndex] == next)
            mergedInside.add(next);
          else stat.remove(next, 1);
        }
      }
      return new CherryOptimizationSubset(first.bds, clause, mergedInside.toArray(), false, first.all, stat, first.initialCardinality);
    }
    throw new RuntimeException("Never happen");
  }
}
