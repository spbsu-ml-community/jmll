package com.spbsu.ml.methods.multilabel;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.multilabel.ClassicMultiLabelLoss;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multiclass.MultiClassOneVsRest;
import com.spbsu.ml.models.multiclass.JoinedBinClassModel;
import com.spbsu.ml.models.multilabel.ThresholdMultiLabelModel;

/**
 * User: qdeee
 * Date: 22.03.15
 */
public class MultiLabelOneVsRest implements VecOptimization<ClassicMultiLabelLoss> {
  private final VecOptimization<LLLogit> weak;

  public MultiLabelOneVsRest(final VecOptimization<LLLogit> weak) {
    this.weak = weak;
  }

  @Override
  public ThresholdMultiLabelModel fit(final VecDataSet learn, final ClassicMultiLabelLoss multiLabelLoss) {
    final Mx targets = multiLabelLoss.getTargets();
    final Func[] result = new Func[targets.columns()];
    for (int j = 0; j < targets.columns(); j++) {
      final Vec target = targets.col(j);
      final LLLogit llLogit = new LLLogit(target, learn);
      final Trans model = weak.fit(learn, llLogit);
      result[j] = MultiClassOneVsRest.extractFunc(model);
    }
    return new ThresholdMultiLabelModel(new JoinedBinClassModel(result), 0.5);
  }
}
