package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.ml.Model;
import com.spbsu.ml.MultiClassModel;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.loss.LogLikelihoodSigmoid;
import com.spbsu.ml.models.AdditiveMultiClassModel;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
* User: solar
* Date: 21.12.2010
* Time: 22:13:54
*/
public class MulticlassBoosting extends ProgressOwner implements MLMultiClassMethodOrder1 {
  MLMultiClassMethodOrder1 weak;
  int iterationsCount;
  double step;
  private final Random rnd;

  public MulticlassBoosting(MLMultiClassMethodOrder1 weak, int iterationsCount, double step, Random rnd) {
    this.weak = weak;
    this.iterationsCount = iterationsCount;
    this.step = step;
    this.rnd = rnd;
  }

  @Override
  public MultiClassModel fit(DataSet learn, Oracle1 loss) {
    final Vec[] points = new Vec[DataTools.countClasses(learn.target())];
    for (int i = 0; i < points.length; i++) {
      points[i] = new ArrayVec(learn.power());
    }
    return fit(learn, loss, points);
  }

  @Override
  public Model fit(DataSet learn, Oracle1 loss, Vec start) {
    throw new UnsupportedOperationException("For multiclass methods continuation from fixed point is not supported");
  }

  public MultiClassModel fit(DataSet learn, Oracle1 loss, Vec[] start) {
    final Vec[] points = start;
    final List<MultiClassModel> models = new LinkedList<MultiClassModel>();
    final AdditiveMultiClassModel result = new AdditiveMultiClassModel(models, step);

    for (int i = 0; i < iterationsCount; i++) {
      final LogLikelihoodSigmoid lls = new LogLikelihoodSigmoid(learn.target());
      final DataSet sampling = DataTools.bootstrap(learn, rnd);
      final MultiClassModel weakModel = weak.fit(sampling, lls, points);
      if (weakModel == null)
        break;

      models.add(weakModel);
      processProgress(result);
      for (int c = 0; c < points.length; c++) {
        final DSIterator it = learn.iterator();
        final Vec point = points[c];
        for (int t = 0; it.advance() && t < points[0].dim(); t++) {
          double val = weakModel.value(it.x(), c);
          point.adjust(t, step * val);
        }
      }
    }

    return result;
  }
}
