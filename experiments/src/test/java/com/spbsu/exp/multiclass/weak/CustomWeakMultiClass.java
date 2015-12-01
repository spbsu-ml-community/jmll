package com.spbsu.exp.multiclass.weak;

import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.FakePool;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.MultiClass;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.models.multiclass.MCModel;

/**
* User: qdeee
* Date: 16.08.14
*/
public class CustomWeakMultiClass extends VecOptimization.Stub<BlockwiseMLLLogit> {
  private final int iters;
  private final double step;

  public CustomWeakMultiClass(int iters, double step) {
    this.iters = iters;
    this.step = step;
  }

  @Override
  public Trans fit(final VecDataSet learnData, final BlockwiseMLLLogit loss) {
    final BFGrid grid = GridTools.medianGrid(learnData, 32);
    final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), SatL2.class), iters, step);

    final IntSeq intTarget = loss.labels();
    final FakePool ds = new FakePool(learnData.data(), intTarget);

    System.out.println(prepareComment(intTarget));
    final ProgressHandler calcer = new ProgressHandler() {
      int iter = 0;

      @Override
      public void invoke(Trans partial) {
        if ((iter + 1) % 20 == 0) {
          final FuncJoin internModel = MCTools.joinBoostingResult((Ensemble) partial);
          final MultiClassModel multiClassModel = new MultiClassModel(internModel);
          final Mx x = internModel.transAll(learnData.data());
          System.out.println("iter=" + iter + ", [learn]MLLLogitValue=" + String.format("%.10f", loss.value(x)) + ", stats=" + MCTools.evalModel(multiClassModel, ds, "[LEARN]", true) + "\r");

        }
        iter++;
      }
    };
//    boosting.addListener(calcer);
    final Ensemble ensemble = boosting.fit(learnData, loss);
    System.out.println();
    final MCModel model = new MultiClassModel(MCTools.joinBoostingResult(ensemble));
    return model;
  }

  private static String prepareComment(final IntSeq labels) {
    final StringBuilder builder = new StringBuilder("Class entries count: { ");
    final int countClasses = MCTools.countClasses(labels);
    for (int i = 0; i < countClasses; i++) {
      builder.append(i)
          .append(" : ")
          .append(MCTools.classEntriesCount(labels, i))
          .append(", ");
    }
    return builder.append("}").toString();
  }
}
