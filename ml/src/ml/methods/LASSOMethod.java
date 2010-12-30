package ml.methods;

import ml.Model;
import ml.data.DSIterator;
import ml.data.DataEntry;
import ml.data.DataSet;
import ml.loss.L2Loss;
import ml.loss.LossFunction;

import java.util.Arrays;
import java.util.Comparator;

/**
 * User: solar
 * Date: 27.12.10
 * Time: 18:04
 */
public class LASSOMethod implements MLMethod {
    private final int iterations;
    private final double step;

    public LASSOMethod(int iterations, double step) {
        this.iterations = iterations;
        this.step = step;
    }

    public Model fit(DataSet learn, LossFunction loss) {
        if (loss.getClass() != L2Loss.class)
            throw new IllegalArgumentException("LASSO can not be applied to loss other than l2");
        final double[] betas = new double[learn.featureCount()];

        double[] values = new double[learn.power()];
        double score = 0;
        {
            final DSIterator it = learn.iterator();
            for (int i = 0; i < values.length; i++) {
                it.advance();
                values[i] = it.y();
                score += it.y() * it.y();
            }
        }

        for (int t = 0; t < iterations; t++) {
            int bestDirection = 0;
            double sign = 0;
            {
                double[] correlations = new double[learn.featureCount()];
                final DSIterator it = learn.iterator();
                int index = 0;
                while (it.advance()) {
                    for (int i = 0; i < correlations.length; i++) {
                        correlations[i] += it.x(i) * values[index];
                    }
                    index++;
                }
                double corr = Math.abs(correlations[0]);
                for (int i = 1; i < correlations.length; i++) {
                    final double current = Math.abs(correlations[i]);
                    if (corr < current) {
                        corr = current;
                        bestDirection = i;
                        sign = Math.signum(correlations[i]);
                    }
                }
            }
            final double signedStep = step * sign;
            betas[bestDirection] += signedStep;
            {
                final DSIterator it = learn.iterator();
                score = 0;
                for (int i = 0; i < values.length; i++) {
                    it.advance();
                    values[i] -= signedStep * it.x(bestDirection);
                    score += values[i] * values[i];
                }
            }
        }
        final double learnScore = score;
        return new Model() {
            public double value(DataEntry point) {
                double result = 0;
                for (int i = 0; i < betas.length; i++) {
                    result += point.x(i) * betas[i];
                }
                return result;
            }

            public double learnScore() {
                return learnScore;
            }

            public String toString() {
                String result = "";
                Integer[] order = new Integer[betas.length];
                for (int k = 0; k < betas.length; k++) {
                    order[k] = k;
                }
                Arrays.sort(order, new Comparator<Integer>() {
                    public int compare(Integer a, Integer b) {
                        return (int)Math.signum(Math.abs(betas[b]) - Math.abs(betas[a]));
                    }
                });
                for (int i = 0; i < betas.length; i++) {
                    if(betas[order[i]] == 0)
                        continue;
                    result += "\t" + order[i] + ": " + betas[order[i]] + "\n";
                }
                return result;
            }
        };
    }
}
