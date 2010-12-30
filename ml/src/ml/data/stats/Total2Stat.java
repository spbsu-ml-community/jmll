package ml.data.stats;

import ml.data.DSIterator;
import ml.data.DataSet;
import ml.data.StatisticCalculator;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 19:19
 */
public class Total2Stat implements StatisticCalculator {
    public double value(DataSet set) {
        final DSIterator iter = set.iterator();
        double sum = 0;
        while(iter.advance()) {
            sum += iter.y() * iter.y();
        }
        return sum;
    }
}
