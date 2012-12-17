package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.idxtrans.ArrayPermutation;
import com.spbsu.ml.Model;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.stats.OrderByFeature;
import com.spbsu.ml.data.stats.Total2Stat;
import com.spbsu.ml.data.stats.TotalStat;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.loss.LossFunction;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:47:15
 */
public class BestAtomicSplitMethod extends ParallelByFeatureMethod {
  public AtomicSplit fit(final DataSet l, final LossFunction loss, FeatureFilter filter) {
    DataSet learn = l;
    assert loss.getClass() == L2Loss.class;
    double minLoss = Double.MAX_VALUE;
    AtomicSplit bestModel = new AtomicSplit(0,0,0,0,minLoss);
    final double total = learn.cache(TotalStat.class);
    final double total2 = learn.cache(Total2Stat.class);
    final int featureCount = learn.xdim();
    final int count = learn.power();
    final OrderByFeature byFeature = learn.cache(OrderByFeature.class);
    for (int f = 0; f < featureCount; f++) {
      if (!filter.relevant(f))
        continue;
      int index = 0;
      double sum = 0;
      final ArrayPermutation order = byFeature.orderBy(f);
      while (index < count) {
        double condition = learn.data().get(order.forward(index));
        sum += learn.target().get(order.forward(index));
        index++;
        while (index < count && learn.data().get(order.forward(index)) <= condition) {
          sum += learn.target().get(order.forward(index));
          index++;
        }
        final double compSum = total - sum;
        final int compCount = count - index;
        final double benefitFromLeft = sum * sum/index - regularize(index);
        final double benefitFromRight = compCount > 0 ? compSum * compSum/compCount - regularize(compCount) : 0;
        double score = total2;
        double vLeft = 0;
        double vRight = 0;
        if (benefitFromLeft > 0) {
          score -= benefitFromLeft;
          vLeft = sum/index;
        }
        if (benefitFromRight > 0) {
          score -= benefitFromRight;
          vRight = compSum/compCount;
        }
        if (score < minLoss) {
          minLoss = score;
          bestModel.feature = f;
          bestModel.condition = condition;
          bestModel.valueLeft = vLeft;
          bestModel.valueRight = vRight;
          bestModel.score = score;
        }
      }
    }
    return bestModel.isCorrect() ? bestModel : null;
  }

  public class AtomicSplit implements Model {
    int feature;
    double condition;
    double valueLeft, valueRight;
    double score;

    AtomicSplit(int feature, double condition, double valueLeft, double valueRight, double score) {
      this.feature = feature;
      this.condition = condition;
      this.valueLeft = valueLeft;
      this.valueRight = valueRight;
      this.score = score;
    }

    public double value(Vec point) {
      if (point.get(feature) <= condition)
        return valueLeft;
      return valueRight;
    }

    public boolean isCorrect() {
      return valueLeft != 0 || valueRight != 0;
    }
  }

  protected double regularize(int count) {
    return 0.01 * Math.log(2)/Math.log(count + 2);
//        return 0.01;
//        return 0;
  }
}
