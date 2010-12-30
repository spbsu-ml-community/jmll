package ml.loss;

import ml.Model;
import ml.data.DSIterator;
import ml.data.DataSet;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:37:55
 */
public class L2Loss implements LossFunction {
    public double[] gradient(double[] point, DataSet learn) {
        double[] result = new double[learn.power()];
        final DSIterator it = learn.iterator();
        int index = 0;
        while(it.advance()) {
            result[index] = - (point[index] - it.y());
            index++;
        }
        return result;
    }

    public double value(Model model, DataSet set) {
        double loss = 0;
        final int count = set.power();
        final DSIterator it = set.iterator();
        while(it.advance()) {
            double v = model.value(it) - it.y();
            loss += v * v;
        }
        return Math.sqrt(loss/count);
    }
}
