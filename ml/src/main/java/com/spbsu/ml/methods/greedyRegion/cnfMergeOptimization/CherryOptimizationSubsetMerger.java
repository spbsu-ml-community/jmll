package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.ml.data.impl.BinarizedDataSet;
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
    final BinarizedDataSet bds = first.bds;
    final AdditiveStatistics stat = factory.create();
    if (first.minimumIndices.length > second.minimumIndices.length) {
      final CherryOptimizationSubset tmp = first;
      first = second;
      second = tmp;
    }
    stat.append(first.stat);
    if (first.isMinimumOutside) {
      final TIntArrayList mergedOutside = new TIntArrayList(first.minimumIndices.length);
      final int[] outside = first.minimumIndices;
      for(int i = 0; i < outside.length; i++) {
        if (second.clause.value(bds, outside[i]) != 1.) {
          mergedOutside.add(outside[i]);
        } else {
          stat.append(outside[i], 1);
        }
      }
      return new CherryOptimizationSubset(first.bds, clause, mergedOutside.toArray(), true, first.all, stat);
    }
    else {
      final int[] secondInside = second.inside();
      final int[] firstInside = first.minimumIndices;
      stat.append(second.stat);
      int left =0;
      int right = 0;
      final TIntArrayList mergedInside = new TIntArrayList(secondInside.length + firstInside.length);
      while (left < firstInside.length && right < secondInside.length) {
        if (firstInside[left] == secondInside[right]) {
          stat.remove(firstInside[left],1);
          mergedInside.add(firstInside[left]);
          ++left;
          ++right;
        }  else if (firstInside[left] < secondInside[right]) {
          mergedInside.add(firstInside[left]);
          ++left;
        } else {
          mergedInside.add(secondInside[right]);
          ++right;
        }
      }
      while (left < firstInside.length) {
        mergedInside.add(firstInside[left]);
        ++left;
      }
      while (right < secondInside.length) {
        mergedInside.add(secondInside[right]);
        ++right;
      }

      return new CherryOptimizationSubset(first.bds, clause, mergedInside.toArray(), false, first.all, stat);
    }
  }


}
