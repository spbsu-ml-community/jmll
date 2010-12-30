package ml.methods;

import ml.Model;
import ml.data.DSIterator;
import ml.data.DataSet;
import ml.data.DataEntry;
import ml.data.stats.Total2Stat;
import ml.data.stats.TotalStat;
import ml.loss.LossFunction;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:47:15
 */
public class BestAtomicSplitMethod extends ParallelByFeatureMethod {
    public Model fit(final DataSet learn, final LossFunction loss, FeatureFilter filter) {
        double minLoss = Double.MAX_VALUE;
        AtomicSplit bestModel = new AtomicSplit(0,0,0,0,minLoss);
        final double total = learn.statistic(TotalStat.class);
        final double total2 = learn.statistic(Total2Stat.class);
        final int featureCount = learn.featureCount();
        final int count = learn.power();
        for (int f = 0; f < featureCount; f++) {
            if (!filter.relevant(f))
                continue;
            int index = 0;
            double sum = 0;
            final DSIterator order = learn.orderBy(f);

            order.advance();
            while (index < count) {
                double condition = order.x(f);
                sum += order.y();
                index++;
                while (order.advance() && order.x(f) <= condition) {
                    sum += order.y();
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

        public double value(DataEntry point) {
            if (point.x(feature) <= condition)
                return valueLeft;
            return valueRight;
        }

        public double learnScore() {
            return score;
        }

        public boolean isCorrect() {
            return valueLeft != 0 || valueRight != 0;
        }
    }

    protected double regularize(int count) {
        return 0.1 * Math.log(2)/Math.log(count + 1);
//        return 0.01;
//        return 0;
    }
}
