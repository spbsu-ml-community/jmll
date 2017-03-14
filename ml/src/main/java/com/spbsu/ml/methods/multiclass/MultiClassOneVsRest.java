package com.spbsu.ml.methods.multiclass;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.multiclass.ClassicMulticlassLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.JoinedBinClassModel;

/**
 * User: qdeee
 * Date: 22.01.15
 */
public class MultiClassOneVsRest implements VecOptimization<ClassicMulticlassLoss> {
  private final VecOptimization<LLLogit> learner;

  public MultiClassOneVsRest(final VecOptimization<LLLogit> learner) {
    this.learner = learner;
  }

  @Override
  public Trans fit(final VecDataSet learn, final ClassicMulticlassLoss multiclassLoss) {
    final IntSeq labels = multiclassLoss.labels();
    final int countClasses = MCTools.countClasses(labels);

    final Func[] models = new Func[countClasses];
    for (int c = 0; c < countClasses; c++) {
      final Vec oneVsRestTarget = MCTools.extractClassForBinary(labels, c);
      final LLLogit llLogit = new LLLogit(oneVsRestTarget, learn.parent());
      final Trans model = learner.fit(learn, llLogit);

      models[c] = extractFunc(model);
    }

    return new JoinedBinClassModel(models);
  }

  public static Func extractFunc(final Trans model) {
    if (model instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) model;
      if (ensemble.last() instanceof Func) {
        return new FuncEnsemble<>(
            ArrayTools.map(ensemble.models, Func.class, new Computable<Trans, Func>() {
              @Override
              public Func compute(final Trans argument) {
                return (Func) argument;
              }
            }),
            ensemble.weights);
      } else {
        throw new IllegalArgumentException("Ensemble doesn't contain a Func");
      }
    } else if (model instanceof Func) {
      return (Func) model;
    } else {
      throw new IllegalArgumentException("Model doesn't contain a Func");
    }
  }
}
