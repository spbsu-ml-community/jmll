package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.models.AdditiveModel;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import static com.spbsu.commons.math.vectors.VecTools.copy;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:13:54
 */
public class Boosting extends ProgressOwner implements MLMethodOrder1 {
  protected final MLMethodOrder1 weak;
  private final Oracle1 weakTarget;
  int iterationsCount;
  double step;
  protected final Random rnd;

  public Boosting(MLMethodOrder1 weak, Oracle1 weakTarget, int iterationsCount, double step, Random rnd) {
    this.weak = weak;
    this.weakTarget = weakTarget;
    this.iterationsCount = iterationsCount;
    this.step = step;
    this.rnd = rnd;
  }

  public Model fit(DataSet learn, Oracle1 loss, Vec start) {
    final List<Model> models = new LinkedList<Model>();
    final AdditiveModel result = new AdditiveModel(models, step);

    Vec point = copy(start);

    for (int i = 0; i < iterationsCount; i++) {
      final Bootstrap sampling = DataTools.bootstrap(learn, rnd);
      Model weakModel = weak.fit(sampling, weakTarget, point);
      if (weakModel == null)
        break;

      models.add(weakModel);
      processProgress(result);
      final DSIterator it = learn.iterator();
      while (it.advance()) {
        double val = weakModel.value(it.x());
        point.adjust(it.index(), step * val);
      }
    }

    return result;
  }

  @Override
  public Model fit(DataSet learn, Oracle1 loss) {
    return fit(learn, loss, new ArrayVec(learn.power()));
  }
}
