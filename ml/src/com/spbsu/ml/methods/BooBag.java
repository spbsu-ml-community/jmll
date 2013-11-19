package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.CursorOracle;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.models.AdditiveModel;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:13:54
 */
public class BooBag<LocalLoss extends Oracle1> extends Boosting<LocalLoss> {
  private final int trees;

  public BooBag(MLMethod<LocalLoss> weak, int iterationsCount, int trees, double step, Random rnd) {
    super(weak, iterationsCount, step, rnd);
    this.trees = trees;
  }

  public Model fit(DataSet learn, CursorOracle<LocalLoss> loss) {
    final List<Model> models = new LinkedList<Model>();
    final AdditiveModel<Model> result = new AdditiveModel<Model>(models, step);

    for (int i = 0; i < iterationsCount; i++) {
      final LocalLoss l2Loss = loss.local();
      final List<Model> modelsForForest = new ArrayList<Model>(trees);
      final AdditiveModel<Model> forest = new AdditiveModel<Model>(modelsForForest, 1./trees);

      for (int t = 0; t < trees; t++) {
        final DataSet gradients = DataTools.bootstrap(learn, rnd);
        Model weakModel = weak.fit(gradients, l2Loss);
        if (weakModel != null)
          modelsForForest.add(weakModel);
      }
      if (modelsForForest.isEmpty())
        break;

      final Vec increment = forest.value(learn);
      VecTools.scale(increment, step);
      loss.moveTo(VecTools.append(loss.cursor()));
      models.add(forest);
      invoke(result);
    }

    return result;
  }
}
