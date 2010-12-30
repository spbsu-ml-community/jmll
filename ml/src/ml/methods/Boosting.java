package ml.methods;

import ml.Model;
import ml.ProgressHandler;
import ml.data.DataEntry;
import ml.data.DSIterator;
import ml.data.DataSet;
import ml.data.DataTools;
import ml.loss.L2Loss;
import ml.loss.LossFunction;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:13:54
 */
public class Boosting implements MLMethod {
    MLMethod weak;
    int iterationsCount;
    double step;
    private ProgressHandler progress;

    public Boosting(MLMethod weak, int iterationsCount, double step) {
        this.weak = weak;
        this.iterationsCount = iterationsCount;
        this.step = step;
    }

    public Model fit(DataSet learn, LossFunction loss) {
        final List<Model> models = new LinkedList<Model>();
        final AdditiveModel result = new AdditiveModel(models);

        double[] point = new double[learn.power()];

        for (int i = 0; i < iterationsCount; i++) {
            final DataSet gradients = DataTools.bootstrap(DataTools.changeTarget(learn, loss.gradient(point, learn)));
            final L2Loss l2Loss = new L2Loss();
            Model weakModel = weak.fit(gradients, l2Loss);
            if (weakModel == null)
                break;
//            System.out.println("" + l2Loss.value(weakModel, learn));
            models.add(weakModel);
            progress.progress(result);
            final DSIterator it = learn.iterator();
            for (int t = 0; t < point.length; t++) {
                it.advance();
                point[t] += step * weakModel.value(it);
            }
        }

        return result;
    }

    public void setProgressHandler(ProgressHandler handler) {
        this.progress = handler;
    }
    private class AdditiveModel implements Model {
        private final List<Model> models;

        public AdditiveModel(List<Model> models) {
            this.models = models;
        }

        public double value(DataEntry point) {
            Iterator<Model> iter = models.iterator();
            double result = 0;
            while (iter.hasNext()) {
                result += step * iter.next().value(point);
            }
            return result;
        }

        public double learnScore() {
            throw new UnsupportedOperationException("Have no idea how to aggregate scores :(");
        }
    }
}
