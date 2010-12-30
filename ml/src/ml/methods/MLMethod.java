package ml.methods;

import ml.Model;
import ml.data.DataSet;
import ml.loss.LossFunction;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:14:38
 */
public interface MLMethod {
    Model fit(DataSet learn, LossFunction loss);
}
