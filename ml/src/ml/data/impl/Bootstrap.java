package ml.data.impl;

import ml.data.DSIterator;
import ml.data.DataSet;
import ml.data.StatisticCalculator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 18:05
 */
public class Bootstrap implements DataSetInner {
    final DataSetImpl parent;
    int[] counts;

    public Bootstrap(DataSetImpl parent) {
        this.parent = parent;
        Random rnd = new Random();
        counts = new int[parent.power()];
        for (int i = 0; i < parent.power(); i++) {
            final int nextIndex = rnd.nextInt(parent.power());
            counts[nextIndex]++;
        }
    }

    public int power() {
        return parent.power();
    }

    public int featureCount() {
        return parent.featureCount();
    }

    public DSIterator iterator() {
        return new DSIterator() {
            int index;
            int pass;
            int offset;
            final double[] data = parent.data(), target = parent.target();
            final int step = data.length / target.length;
            public boolean advance() {
                if (index < target.length && --pass > 0)
                    return true;
                while (++index < target.length && counts[index] == 0){}
                if (index >= target.length)
                    return false;
                pass = counts[index];
                offset = index * step;
                return true;
            }

            public double y() {
                return target[offset/step];
            }

            public double x(int i) {
                return data[offset + i];
            }
        };
    }

    public DSIterator orderBy(int featureIndex) {
        final int[] order = parent.order(featureIndex);
        return new DSIterator() {
            int index = -1;
            int pass = 0;
            int offset;
            final double[] data = parent.data(), target = parent.target();
            final int step = data.length / target.length;
            public boolean advance() {
                if (index < target.length && --pass > 0)
                    return true;
                while (++index < order.length && counts[order[index]] == 0){}
                if (index >= order.length)
                    return false;
                pass = counts[order[index]];
                offset = order[index] * step;
                return true;
            }

            public double y() {
                return target[offset/step];
            }

            public double x(int i) {
                return data[offset + i];
            }
        };
    }

    Map<Class<? extends StatisticCalculator>, Double> statistics= new HashMap<Class<? extends StatisticCalculator>, Double>();
    public synchronized double statistic(Class<? extends StatisticCalculator> type) {
        Double result = statistics.get(type);
        if (result == null) {
            try {
                final StatisticCalculator calculator = type.newInstance();
                result = calculator.value(this);

            } catch (Exception e) {
                e.printStackTrace(); // I'd be extremely surprised if this happen :)
            }

        }
        return result;
    }

    public DataSet bootstrap() {
        throw new IllegalArgumentException("Bootstap for already bootrapped set is not supported");
    }

    public double[] target() {
        throw new NotImplementedException();
    }

    public double[] data() {
        throw new NotImplementedException();
    }
}
