package com.expleague.exp.multiclass.weak;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.*;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.FakePool;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.BFGrid;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.MultiClass;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.MultiClassModel;
import com.expleague.ml.models.multiclass.MCModel;

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
    final FakePool ds = FakePool.create(learnData.data(), intTarget);

    System.out.println(prepareComment(intTarget));
    final ProgressHandler calcer = new ProgressHandler() {
      int iter = 0;

      @Override
      public void accept(Trans partial) {
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
