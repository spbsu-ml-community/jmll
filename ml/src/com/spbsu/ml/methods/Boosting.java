package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.models.AdditiveModel;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:13:54
 */
public class Boosting implements MLMethodOrder1 {
  MLMethodOrder1 weak;
  int iterationsCount;
  double step;
  private List<WeakReference<ProgressHandler>> progress = new ArrayList<WeakReference<ProgressHandler>>();

  public Boosting(MLMethodOrder1 weak, int iterationsCount, double step) {
    this.weak = weak;
    this.iterationsCount = iterationsCount;
    this.step = step;
  }

  public Model fit(DataSet learn, Oracle1 loss) {
    final List<Model> models = new LinkedList<Model>();
    final AdditiveModel result = new AdditiveModel(models, step);

    Vec point = new ArrayVec(learn.power());

    for (int i = 0; i < iterationsCount; i++) {
      final Vec gradient = loss.gradient(point);
      final DataSet gradients = DataTools.bootstrap(DataTools.changeTarget(learn, loss.gradient(point)));
      final L2Loss l2Loss = new L2Loss(gradient);
      Model weakModel = weak.fit(gradients, l2Loss);
      if (weakModel == null)
        break;

      models.add(weakModel);
      final Iterator<WeakReference<ProgressHandler>> progIter = progress.iterator();
      while (progIter.hasNext()) {
        WeakReference<ProgressHandler> next = progIter.next();
        final ProgressHandler progressHandler;
        if ((progressHandler = next.get()) != null) {
          progressHandler.progress(result);
        }
        else progIter.remove();
      }
      final DSIterator it = learn.iterator();
      for (int t = 0; it.advance() && t < point.dim(); t++) {
        double val = weakModel.value(it.x());
        point.adjust(t, step * val);
      }
    }

    return result;
  }

  public void addProgressHandler(ProgressHandler handler) {
    this.progress.add(new WeakReference<ProgressHandler>(handler));
  }
}
