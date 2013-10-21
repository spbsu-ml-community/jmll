package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.ChangedTarget;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.models.AdditiveModel;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:13:54
 */
public class BooBag extends Boosting {
  private final int trees;

  public BooBag(MLMethodOrder1 weak, int iterationsCount, int trees, double step, Random rnd) {
    super(weak, null, iterationsCount, step, rnd);
    this.trees = trees;
  }

  public Model fit(DataSet learn, Oracle1 loss, Vec start) {
    final List<Model> models = new LinkedList<Model>();
    final AdditiveModel result = new AdditiveModel(models, step);

    Vec point = copy(start);
    Vec regressionStartPoint = new ArrayVec(point.dim());

    for (int i = 0; i < iterationsCount; i++) {
      final Vec gradient = loss.gradient(point);
      final ChangedTarget gradientOrig = DataTools.changeTarget(learn, loss.gradient(point));
      final L2Loss l2Loss = new L2Loss(gradient);
      final List<Model> modelsForForest = new ArrayList<Model>(trees);
      final AdditiveModel forest = new AdditiveModel(modelsForForest, 1./trees);

      for (int t = 0; t < trees; t++) {
        final DataSet gradients = DataTools.bootstrap(gradientOrig, rnd);
        Model weakModel = weak.fit(gradients, l2Loss, regressionStartPoint);
        if (weakModel != null)
          modelsForForest.add(weakModel);
      }
      if (modelsForForest.isEmpty())
        break;
      models.add(forest);
      processProgress(result);
      final DSIterator it = learn.iterator();
      for (int u = 0; it.advance() && u < point.dim(); u++) {
        double val = forest.value(it.x());
        point.adjust(u, step * val);
      }
    }

    return result;
  }
}
