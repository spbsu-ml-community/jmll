package com.spbsu.ml.cli.output.printers;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.Func;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multiclass.util.ConfusionMatrix;
import com.spbsu.ml.models.MCModel;
import com.spbsu.ml.models.MultiClassModel;

/**
 * User: qdeee
 * Date: 04.09.14
 */
public class ResultsPrinter {
  public static void printResults(final Computable computable, final Pool<?> learn, final Pool<?> test, final Func loss, final Func[] metrics) {
    System.out.print("Learn: " + loss.value(DataTools.calcAll(computable, learn.vecData())) + " Test:");
    for (final Func metric : metrics) {
      System.out.print(" " + metric.value(DataTools.calcAll(computable, test.vecData())));
    }
    System.out.println();
  }

  public static void printMulticlassResults(final Computable computable, final Pool<?> learn, final Pool<?> test) {
    final MCModel mcModel;
    if (computable instanceof Ensemble && ((Ensemble) computable).last() instanceof FuncJoin) {
      final FuncJoin funcJoin = MCTools.joinBoostingResult((Ensemble) computable);
      mcModel = new MultiClassModel(funcJoin);
    } else if (computable instanceof MCModel) {
      mcModel = (MCModel) computable;
    } else return;

    final IntSeq learnTarget = learn.target(BlockwiseMLLLogit.class).labels();
    final Vec learnPredict = mcModel.bestClassAll(learn.vecData().data());
    final ConfusionMatrix learnConfusionMatrix = new ConfusionMatrix(learnTarget, VecTools.toIntSeq(learnPredict));
    System.out.println("LEARN:");
    System.out.println(learnConfusionMatrix.toSummaryString());
    System.out.println(learnConfusionMatrix.toClassDetailsString());

    final IntSeq testTarget = test.target(BlockwiseMLLLogit.class).labels();
    final Vec testPredict = mcModel.bestClassAll(test.vecData().data());
    final ConfusionMatrix testConfusionMatrix = new ConfusionMatrix(testTarget, VecTools.toIntSeq(testPredict));
    System.out.println("TEST:");
    System.out.println(testConfusionMatrix.toSummaryString());
    System.out.println(testConfusionMatrix.toClassDetailsString());
    System.out.println();
  }
}
