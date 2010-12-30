package ml.methods;

import Jama.Matrix;
import ml.Model;
import ml.data.DSIterator;
import ml.data.DataEntry;
import ml.data.DataSet;
import ml.data.impl.DataSetImpl;
import ml.data.impl.NormalizedDataSet;
import ml.loss.L2Loss;
import ml.loss.LossFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * User: solar
 * Date: 27.12.10
 * Time: 18:04
 */
public class LARSMethod implements MLMethod {
    private static final double E = 1e-7;
    private class Direction {
        double sign;
        int index;
        private Direction(double sign, int index) {
            this.sign = sign;
            this.index = index;
        }
    }

    public Model fit(DataSet orig, LossFunction loss) {
        if (loss.getClass() != L2Loss.class)
            throw new IllegalArgumentException("LASSO can not be applied to loss other than l2");
        final int featuresCount = orig.featureCount();
        final double[] betas = new double[featuresCount];
        double[] values = new double[orig.power()];
        double score = 0;
        NormalizedDataSet learn = new NormalizedDataSet((DataSetImpl)orig);
        {
            final DSIterator it = learn.iterator();
            for (int i = 0; i < values.length; i++) {
                it.advance();
                values[i] = it.y();
                score += it.y() * it.y();
            }
        }

        for (int t = 0; t < featuresCount; t++) {
            double[] correlations = new double[featuresCount];
            double bestCorr;
            final List<Direction> selectedDirections = new ArrayList<Direction>(featuresCount);
            {
                final DSIterator it = learn.iterator();
                int index = 0;
                while (it.advance()) {
                    for (int i = 0; i < correlations.length; i++) {
                        correlations[i] += it.x(i) * values[index];
                    }
                    index++;
                }
                bestCorr = Math.abs(correlations[0]);
                selectedDirections.add(new Direction(Math.signum(correlations[0]), 0));
                for (int i = 1; i < correlations.length; i++) {
                    final double current = Math.abs(correlations[i]);
                    final double diff = current - bestCorr;
                    if (diff > E) {
                        bestCorr = current;
                        selectedDirections.clear();
                        selectedDirections.add(new Direction(Math.signum(correlations[i]), i));
                    }
                    else if (diff < E && diff > -E) {
                        selectedDirections.add(new Direction(Math.signum(correlations[i]), i));
                    }
                }
            }
            final Matrix inverseCo;
            {
                final Matrix covariance = new Matrix(selectedDirections.size(), selectedDirections.size());
                final DSIterator it = learn.iterator();
                while (it.advance()) {
                    for (int i = 0; i < selectedDirections.size(); i++) {
                        final Direction d1 = selectedDirections.get(i);
                        for (int j = 0; j < selectedDirections.size(); j++) {
                            final Direction d2 = selectedDirections.get(j);
                            covariance.set(i, j, covariance.get(i, j) + it.x(d1.index) * it.x(d2.index));
                        }
                    }
                }
                inverseCo = covariance.inverse();
            }
            final Matrix vec1 = new Matrix(selectedDirections.size(), 1, 1);
            double norm = Math.sqrt(vec1.transpose().times(inverseCo.times(vec1)).get(0, 0));
            Matrix w = inverseCo.times(vec1).times(norm);
            double[] equiangular = new double[learn.power()];
            {
                final DSIterator it = learn.iterator();
                int index = 0;
                while (it.advance()) {
                    for (int i = 0; i < selectedDirections.size(); i++) {
                        final Direction direction = selectedDirections.get(i);
                        equiangular[index] += direction.sign * it.x(direction.index) * w.get(i, 0);
                    }
                    index++;
                }
            }

            double[] a = new double[featuresCount];
            {
                final DSIterator it = learn.iterator();
                int index = 0;
                while (it.advance()) {
                    final double eqaComponent = equiangular[index];
                    for (int i = 0; i < featuresCount; i++) {
                        a[i] += it.x(i) * eqaComponent;
                    }
                    index++;
                }
            }

            double step = Double.MAX_VALUE;
            {
                for (final Direction direction : selectedDirections) {
                    int j = direction.index;
                    final double s1 = (bestCorr - correlations[j])/(norm - a[j]);
                    final double s2 = (bestCorr + correlations[j])/(norm + a[j]);
                    if (s1 > 0)
                        step = Math.min(s1, step);
                    if (s2 > 0)
                        step = Math.min(s2, step);
                }
            }

            for (final Direction direction : selectedDirections) {
                final double signedStep = step * direction.sign;
                betas[direction.index] += signedStep;
            }
            {
                final DSIterator it = learn.iterator();
                score = 0;
                for (int i = 0; i < values.length; i++) {
                    it.advance();
                    for (final Direction direction : selectedDirections) {
                        final double signedStep = step * direction.sign;
                        values[i] -= signedStep * it.x(direction.index);
                        score += values[i] * values[i];
                    }
                }
            }
        }
        final double learnScore = score;
        final double[] means = learn.means();
        final double[] norms = learn.norms();
        final double meanTarget = learn.meanTarget();
        return new Model() {
            public double value(DataEntry point) {
                double result = 0;
                for (int i = 0; i < betas.length; i++) {
                    if (norms[i] == 0) continue;
                    result += (point.x(i) - means[i]) / norms[i] * betas[i];
                }
                return result + meanTarget;
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
                        return (int) Math.signum(Math.abs(betas[b]) - Math.abs(betas[a]));
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
