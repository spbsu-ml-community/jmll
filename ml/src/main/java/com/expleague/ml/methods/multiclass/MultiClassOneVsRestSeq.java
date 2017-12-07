package com.expleague.ml.methods.multiclass;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.loss.multiclass.ClassicMulticlassLoss;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.models.multiclass.JoinedBinClassModelSeq;

import java.util.function.Function;

public class MultiClassOneVsRestSeq<T> implements SeqOptimization<T, ClassicMulticlassLoss> {
  private final SeqOptimization<T, LLLogit> learner;

  public MultiClassOneVsRestSeq(final SeqOptimization<T, LLLogit> learner) {
    this.learner = learner;
  }

  @Override
  public Function<Seq<T>, Vec> fit(DataSet<Seq<T>> learn,
                                   final ClassicMulticlassLoss multiclassLoss) {
    final IntSeq labels = multiclassLoss.labels();
    final int countClasses = MCTools.countClasses(labels);

    //noinspection unchecked
    final Function<Seq<T>, Vec>[] models = new Function[countClasses];
    for (int c = 0; c < countClasses; c++) {
      final Vec oneVsRestTarget = MCTools.extractClassForBinary(labels, c);
      final LLLogit llLogit = new LLLogit(oneVsRestTarget, learn.parent());
      final Function<Seq<T>, Vec> model = learner.fit(learn, llLogit);

      models[c] = model;
    }
    return new JoinedBinClassModelSeq<>(models);
  }
}
