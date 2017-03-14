package com.spbsu.ml.cli.output.printers;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.MultiLabelTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multiclass.util.ConfusionMatrix;
import com.spbsu.ml.loss.multiclass.util.MultilabelConfusionMatrix;
import com.spbsu.ml.loss.multiclass.util.MultilabelExampleTableOutput;
import com.spbsu.ml.loss.multiclass.util.MultilabelThresholdPrecisionMatrix;
import com.spbsu.ml.loss.multilabel.MultiLabelExactMatch;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.models.multiclass.MCModel;
import com.spbsu.ml.models.multilabel.MultiLabelBinarizedModel;
import com.spbsu.ml.models.multilabel.MultiLabelModel;

/**
 * User: qdeee
 * Date: 04.09.14
 */
@SuppressWarnings("UseOfSystemOutOrSystemErr")
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

  public static void printMultilabelResult(final Computable computable, final Pool<?> learn, final Pool<?> test) {
    final MultiLabelModel mlModel = MultiLabelTools.extractMultiLabelModel((Trans) computable);

    final Mx learnTargets = learn.multiTarget(MultiLabelExactMatch.class).getTargets();
    final Mx learnPredicted = mlModel.predictLabelsAll(learn.vecData().data());
    final MultilabelConfusionMatrix learnConfusionMatrix = new MultilabelConfusionMatrix(learnTargets, learnPredicted);
    System.out.println("[LEARN]");
    System.out.println(learnConfusionMatrix.toSummaryString());
    System.out.println(learnConfusionMatrix.toClassDetailsString());

    final Mx testTargets = test.multiTarget(MultiLabelExactMatch.class).getTargets();
    final Mx testPredicted = mlModel.predictLabelsAll(test.vecData().data());
    final MultilabelConfusionMatrix testConfusionMatrix = new MultilabelConfusionMatrix(testTargets, testPredicted);
    System.out.println("[TEST]");
    System.out.println(testConfusionMatrix.toSummaryString());
    System.out.println(testConfusionMatrix.toClassDetailsString());

    if (mlModel instanceof MultiLabelBinarizedModel) {
      final Trans model = ((MultiLabelBinarizedModel)mlModel).getInternModel();
      final Mx testScores = model.transAll(test.vecData().data());
      System.out.println(new MultilabelThresholdPrecisionMatrix(testScores, testTargets, 100, "=== Precision/recall curve on TEST ===").toThresholdPrecisionMatrix());

      final Mx learnScores = model.transAll(learn.vecData().data());
      System.out.println(new MultilabelThresholdPrecisionMatrix(learnScores, learnTargets, 100, "=== Precision/recall curve on LEARN ===").toThresholdPrecisionMatrix());

      System.out.println(new MultilabelExampleTableOutput(testScores, testTargets, test, "=== Scores for examples on TEST ===\n").toExampleTableMatrix());

      System.out.println(new MultilabelExampleTableOutput(learnScores, learnTargets, learn, "=== Scores for examples on LEARN ===\n").toExampleTableMatrix());
    }
  }
}
