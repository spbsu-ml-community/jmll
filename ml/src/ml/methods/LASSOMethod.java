package ml.methods;

import ml.data.DSIterator;
import ml.data.DataSet;
import ml.loss.L2Loss;
import ml.loss.LossFunction;
import ml.models.LinearModel;

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

    public LinearModel fit(DataSet learn, LossFunction loss) {
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
        return new LinearModel(betas, score);
    }

}
