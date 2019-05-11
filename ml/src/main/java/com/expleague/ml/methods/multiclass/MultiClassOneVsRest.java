package com.expleague.ml.methods.multiclass;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.loss.multiclass.ClassicMulticlassLoss;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.multiclass.JoinedBinClassModel;

import java.util.function.Function;
import java.util.stream.IntStream;

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
        return new FuncEnsemble<>(IntStream.range(0, ensemble.size()).<Trans>mapToObj(ensemble::model).map(f -> (Func)f).toArray(Func[]::new),
            ensemble.weights());
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
