package ml.loss;

import ml.Model;
import ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:25:39
 */
public interface LossFunction {
    double[] gradient(double[] point, DataSet learn);
    double value(Model model, DataSet set);
}
