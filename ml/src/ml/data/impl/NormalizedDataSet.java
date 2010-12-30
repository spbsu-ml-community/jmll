package ml.data.impl;

import ml.data.DSIterator;

/**
 * User: solar
 * Date: 28.12.10
 * Time: 11:34
 */
public class NormalizedDataSet extends DataSetImpl {
    private final DataSetImpl parent;
    private final double meanTarget;
    private final double[] mean;
    private final double[] norm;

    public NormalizedDataSet(DataSetImpl parent) {
        super(parent.data(), parent.target());
        this.parent = parent;
        double targetSum = 0;
        mean = new double[parent.featureCount()];
        norm = new double[parent.featureCount()];
        final DSIterator it = parent.iterator();
        while (it.advance()) {
            targetSum += it.y();
            for (int i = 0; i < mean.length; i++) {
                mean[i] += it.x(i);
                norm[i] += it.x(i) * it.x(i);
            }
        }
        final int power = parent.power();
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= power;
            norm[i] = Math.sqrt(norm[i]);
        }
        meanTarget = targetSum / power;
    }

    int[] order(int fIndex) {
        return parent.order(fIndex);
    }

    protected double map(double value, int index) {
        if (index < 0)
            return (value - meanTarget);
        if (norm[index] == 0)
            return 0;
        return  (value - mean[index]) / norm[index];
    }

    public DSIterator orderBy(int featureIndex) {
        final int[] order = order(featureIndex);
        return new DSIterator() {
            double[] data = data();
            double[] target = target();
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
                return map(target[offset/step], -1);
            }
            public double x(int i) {
                return map(data[offset + i], i);
            }
        };
    }

    public DSIterator iterator() {
        return new DSIterator() {
            double[] data = data();
            double[] target = target();
            final int step = featureCount();
            int index = -step;

            public boolean advance() {
                return (index+=step) < data.length;
            }

            public double y() {
                return map(target[index/step], -1);
            }

            public double x(int i) {
                return map(data[index + i], i);
            }
        };
    }

    public double[] means() {
        return mean;
    }

    public double[] norms() {
        return norm;
    }

    public double meanTarget() {
        return meanTarget;
    }
}
