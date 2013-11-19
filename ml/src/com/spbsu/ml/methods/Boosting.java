package com.spbsu.ml.methods;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.ml.CursorOracle;
import com.spbsu.ml.Model;
import com.spbsu.ml.Oracle0;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.models.AdditiveModel;

import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
* User: solar
* Date: 21.12.2010
* Time: 22:13:54
*/
public class Boosting<LocalLoss extends Oracle0> extends WeakListenerHolderImpl<Model> implements MLMethod<CursorOracle<LocalLoss>> {
  protected final MLMethod<LocalLoss> weak;
  int iterationsCount;
  double step;
  protected final Random rnd;

  public Boosting(MLMethod<LocalLoss> weak, int iterationsCount, double step, Random rnd) {
    this.weak = weak;
    this.iterationsCount = iterationsCount;
    this.step = step;
    this.rnd = rnd;
  }

  public Model fit(DataSet learn, CursorOracle<LocalLoss> loss) {
    final List<Model> models = new LinkedList<Model>();
    final AdditiveModel<Model> result = new AdditiveModel<Model>(models, step);

    for (int i = 0; i < iterationsCount; i++) {
      final Bootstrap sampling = DataTools.bootstrap(learn, rnd);
      Model weakModel = weak.fit(sampling, loss.local());
      if (weakModel == null)
        break;
      final Vec increment = weakModel.value(learn);
      VecTools.scale(increment, step);
      loss.moveTo(VecTools.append(loss.cursor(), increment));
      models.add(weakModel);
      invoke(result);
    }

    return result;
  }
}
