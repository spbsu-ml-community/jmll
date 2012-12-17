package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Model;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.loss.LossFunction;
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
public class Boosting implements MLMethod {
  MLMethod weak;
  int iterationsCount;
  double step;
  private List<WeakReference<ProgressHandler>> progress = new ArrayList<WeakReference<ProgressHandler>>();

  public Boosting(MLMethod weak, int iterationsCount, double step) {
    this.weak = weak;
    this.iterationsCount = iterationsCount;
    this.step = step;
  }

  public Model fit(DataSet learn, LossFunction loss) {
    final List<Model> models = new LinkedList<Model>();
    final AdditiveModel result = new AdditiveModel(models, step);

    Vec point = new ArrayVec(learn.power());

    for (int i = 0; i < iterationsCount; i++) {
      final Vec gradient = loss.gradient(point, learn);
//      final DataSet gradients = DataTools.changeTarget(learn, gradient);
      final DataSet gradients = DataTools.bootstrap(DataTools.changeTarget(learn, loss.gradient(point, learn)));
      final L2Loss l2Loss = new L2Loss();
      Model weakModel = weak.fit(gradients, l2Loss);
      if (weakModel == null)
        break;
//      System.out.println("\npoint: " + VecTools.norm(point) + " grad: " + VecTools.norm(gradient));
//            System.out.println("" + l2Loss.value(weakModel, learn));
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
//      int nz = 0;
      for (int t = 0; it.advance() && t < point.dim(); t++) {
        double val = weakModel.value(it.x());
//        if (Math.abs(val) > 0)
//          nz++;
        point.adjust(t, step * val);
      }
//      System.out.println("\nNon zeroes: " + nz);
    }

    return result;
  }

  public void addProgressHandler(ProgressHandler handler) {
    this.progress.add(new WeakReference<ProgressHandler>(handler));
  }
}
