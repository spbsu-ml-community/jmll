package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.L2Loss;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:13:54
 */
public class MulticlassBoosting extends ProgressOwner implements MLMethodOrder1 {
  MLMethodOrder1 weak;
  int iterationsCount;
  double step;

  public MulticlassBoosting(MLMethodOrder1 weak, int iterationsCount, double step) {
    this.weak = weak;
    this.iterationsCount = iterationsCount;
    this.step = step;
  }

  public Model fit(DataSet learn, Oracle1 loss) {
    final Vec currentTarget = new ArrayVec(learn.power());
    final Vec point = new ArrayVec(learn.power());

    final List<Model> models = new LinkedList<Model>();
    final List<boolean[]> masks = new ArrayList<boolean[]>();
    MulticlassModel<Model> result = new MulticlassModel<Model>(models, masks, step);
    boolean[] prevMask = new boolean[learn.power()];

    for (int i = 0; i < iterationsCount; i++) {
      final Vec gradient = loss.gradient(point);
      final DataSet gradients = DataTools.bootstrap(DataTools.changeTarget(learn, loss.gradient(point)));
      final L2Loss l2Loss = new L2Loss(gradient);
      Model weakModel = weak.fit(gradients, l2Loss);
      if (weakModel == null)
        break;

      models.add(weakModel);
      processProgress(result);
      final DSIterator it = learn.iterator();
      for (int t = 0; it.advance() && t < point.dim(); t++) {
        double val = weakModel.value(it.x());
        point.adjust(t, step * val);
      }
    }

    return result;
  }

  @Override
  public Model fit(DataSet learn, Oracle1 loss, Vec start) {
    return null;  //To change body of implemented methods use File | Settings | File Templates.
  }
}
