package ml.data.impl;

import ml.data.DSIterator;
import ml.data.StatisticCalculator;
import ml.data.DataSet;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

/**
 * User: solar
 * Date: 26.12.10
 * Time: 17:36
 */
public class DataSetImpl implements DataSetInner {
    private final double[] data;
    private final double[] target;

    public DataSetImpl(double[] data, double[] target) {
        this.data = data;
        this.target = target;
    }

    public int power() {
        return target.length;
    }

    public int featureCount() {
        return data.length / target.length;
    }

    public DSIterator iterator() {
        return new DSIterator() {
            double[] data = DataSetImpl.this.data();
            double[] target = DataSetImpl.this.target();
            final int step = featureCount();
            int index = -step;

            public boolean advance() {
                return (index+=step) < data.length;
            }

            public double y() {
                return target[index/step];
            }

            public double x(int i) {
                return data[index + i];
            }
        };
    }

    Map<Integer, int[]> orders = new HashMap<Integer, int[]>();
    synchronized int[] order(final int featureIndex) {
        int[] result = orders.get(featureIndex);
        if (result == null) {
            Integer[] order = new Integer[power()];
            for (int t = 0; t < power(); t++) {
                order[t] = t;
            }
            Arrays.sort(order, new Comparator<Integer>() {
                public int compare(Integer a, Integer b) {
                    return (int) Math.signum(data[a * featureCount() + featureIndex] - data[b * featureCount() + featureIndex]);
                }
            });
            result = new int[power()];
            for (int t = 0; t < result.length; t++) {
                result[t] = order[t];
            }
            orders.put(featureIndex, result);
        }
        return result;
    }

    public DSIterator orderBy(final int featureIndex) {
        final int[] order = order(featureIndex);
        return new DSIterator() {
            double[] data = DataSetImpl.this.data();
            double[] target = DataSetImpl.this.target();
            int index = -1;
            int offset;
            final int step = featureCount();

            public boolean advance() {
                if (++index >= order.length)
                    return false;
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
        return new Bootstrap(this);
    }

    public double[] target() {
        return target;
    }

    public double[] data() {
        return data;
    }
}
