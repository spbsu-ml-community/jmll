package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.ml.methods.greedyMergeOptimization.MergeOptimization;
import com.spbsu.ml.models.CNF;
import gnu.trove.list.array.TIntArrayList;

import java.util.ArrayList;
import java.util.BitSet;

/**
 * Created by noxoomo on 30/11/14.
 */
public class CherryOptimizationSubsetMerger implements MergeOptimization<CherryOptimizationSubset> {
  private final Factory<AdditiveStatistics> factory;

  public CherryOptimizationSubsetMerger(Factory<AdditiveStatistics> factory) {
    this.factory =  factory;
  }

  private CNF.Condition[] merge(CNF.Condition[] leftConditions, CNF.Condition[] rightConditions) {
    int left = 0;
    int right = 0;
    ArrayList<CNF.Condition> conditions = new ArrayList<>(leftConditions.length + rightConditions.length);
    while (left != leftConditions.length && right != rightConditions.length) {
      if (leftConditions[left].feature == rightConditions[right].feature) {
        BitSet used = (BitSet) rightConditions[right].used.clone();
        used.or(leftConditions[left].used);
        CNF.Condition newCondition = new CNF.Condition(rightConditions[right].feature, used);
        ++left;
        ++right;
        conditions.add(newCondition);
        continue;
      }

      if (leftConditions[left].feature < rightConditions[right].feature) {
        BitSet used = (BitSet) leftConditions[left].used.clone();
        CNF.Condition newCondition = new CNF.Condition(leftConditions[left].feature, used);
        ++left;
        conditions.add(newCondition);
      } else {
        BitSet used = (BitSet) rightConditions[right].used.clone();
        CNF.Condition newCondition = new CNF.Condition(rightConditions[right].feature, used);
        ++right;
        conditions.add(newCondition);
      }
    }

    while (left < leftConditions.length) {
      BitSet used = (BitSet) leftConditions[left].used.clone();
      CNF.Condition newCondition = new CNF.Condition(leftConditions[left].feature, used);
      ++left;
      conditions.add(newCondition);
    }
    while (right < rightConditions.length) {
      BitSet used = (BitSet) rightConditions[right].used.clone();
      CNF.Condition newCondition = new CNF.Condition(rightConditions[right].feature, used);
      ++right;
      conditions.add(newCondition);
    }
    return (conditions.toArray(new CNF.Condition[conditions.size()]));

  }


  @Override
  public CherryOptimizationSubset merge(CherryOptimizationSubset first, CherryOptimizationSubset second) {
    final CNF.Condition[] conditions = merge(first.clause.conditions, second.clause.conditions);

    final CNF.Clause clause = new CNF.Clause(first.bds.grid(), conditions);

    final AdditiveStatistics stat = factory.create();
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
          else stat.remove(next, 1);
        }
      }
      return new CherryOptimizationSubset(first.bds, clause, mergedOutside.toArray(), true, first.all, stat);
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

//      AdditiveStatistics test = factory.create();
//      for (int i=0; i < first.all.length;++i) {
//        test.append(first.all[i],1);
//      }
//      for (int j=0; j < mergedOutside.size();++j) {
//        test.remove(mergedOutside.get(j),1);
//      }
      return new CherryOptimizationSubset(first.bds, clause, mergedOutside.toArray(), true, first.all, stat);
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

//      AdditiveStatistics test = factory.create();
//      for (int i=0; i < first.all.length;++i) {
//        test.append(first.all[i],1);
//      }
//      for (int j=0; j < mergedOutside.size();++j) {
//        test.remove(mergedOutside.get(j),1);
//      }

      return new CherryOptimizationSubset(first.bds, clause, mergedOutside.toArray(), true, first.all, stat);
    }
    else if (!first.isMinimumOutside && !second.isMinimumOutside) {
      stat.append(first.stat);
      AdditiveStatistics inside = factory.create();
      final int[] firstInside = first.minimumIndices;
      final int[] secondInside = second.minimumIndices;
      final TIntArrayList mergedInside = new TIntArrayList(second.minimumIndices.length);
      int secondIndex = 0;
      for (int i = 0; i <= firstInside.length; i++) {
        final boolean last = i == firstInside.length;
        final int next = !last ? firstInside[i] : Integer.MAX_VALUE;
        while (secondIndex < secondInside.length && secondInside[secondIndex] < next) {
          mergedInside.add(secondInside[secondIndex]);
          inside.append(secondInside[secondIndex], 1);
          secondIndex++;
        }
        if (!last) {
          inside.append(next, 1);
          mergedInside.add(next);
          if (secondIndex < secondInside.length && secondInside[secondIndex] == next)
            secondIndex++;
        }
      }
//      AdditiveStatistics test = factory.create();
//      for (int i=0; i < mergedInside.size();++i) {
//        test.append(mergedInside.get(i),1);
//      }

//        stat.append(inside);
      return new CherryOptimizationSubset(first.bds, clause, mergedInside.toArray(), false, first.all, inside);
    }
    throw new RuntimeException("Never happen");
  }
}
