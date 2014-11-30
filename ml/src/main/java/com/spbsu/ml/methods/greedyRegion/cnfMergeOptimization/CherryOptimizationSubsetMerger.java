package com.spbsu.ml.methods.greedyRegion.cnfMergeOptimization;

import com.spbsu.commons.func.AdditiveStatistics;
import com.spbsu.commons.func.Factory;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.loss.StatBasedLoss;
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

  public CherryOptimizationSubsetMerger(StatBasedLoss<AdditiveStatistics> loss) {
    this.factory = loss.statsFactory();
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

//  ThreadPoolExecutor exec = ThreadTools.createBGExecutor("merge thread", -1);

  @Override
  public CherryOptimizationSubset merge(CherryOptimizationSubset first, CherryOptimizationSubset second) {
    if (first.outside.length == 0 && second.outside.length == 0) {
      return first;
    }
    if (first.regularization == Double.POSITIVE_INFINITY || second.regularization == Double.POSITIVE_INFINITY) { //skip bad subsets
      return first;
    }
    CNF.Condition[] conditions = merge(first.clause.conditions, second.clause.conditions);


    AdditiveStatistics stat = factory.create();
    if (first.outside.length < second.outside.length) {
      CherryOptimizationSubset tmp = first;
      first = second;
      second = tmp;
    }

    final CNF.Clause clause = new CNF.Clause(first.bds.grid(), conditions);
    final BinarizedDataSet bds = first.bds;
    final boolean inside[] = new boolean[first.outside.length];
    final int[] points = first.outside;
//    final CountDownLatch latch = new CountDownLatch(first.outside.length);
    for (int i = 0; i < points.length; ++i) {
      inside[i] = clause.value(bds, points[i]) == 1.0;
    }

//    for (int i = 0; i < points.length; ++i) {
//      final int fIndex = i;
//      exec.submit(new Runnable() {
//        @Override
//        public void run() {
//          inside[fIndex] = layer.value(bds, points[fIndex]) == 1.0;
//          latch.countDown();
//        }
//      });
//    }
//
//    try {
//      latch.await();
//    } catch (InterruptedException e) {
//      //skip
//    }

    stat.append(first.stat);
    TIntArrayList mergedOutside = new TIntArrayList();
    TIntArrayList mergedInside = new TIntArrayList(first.inside);

    for (int i = 0; i < points.length; ++i) {
      if (inside[i]) {
        stat.append(points[i], 1);
        mergedInside.add(points[i]);
      } else {
        mergedOutside.add(points[i]);
      }
    }
    return new CherryOptimizationSubset(first.bds, clause, mergedInside.toArray(), mergedOutside.toArray(), stat);
  }


}
