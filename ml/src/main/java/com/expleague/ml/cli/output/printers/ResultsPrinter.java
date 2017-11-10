package com.expleague.ml.cli.output.printers;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.MultiLabelTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncJoin;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.loss.multiclass.util.ConfusionMatrix;
import com.expleague.ml.loss.multiclass.util.MultilabelConfusionMatrix;
import com.expleague.ml.loss.multiclass.util.MultilabelExampleTableOutput;
import com.expleague.ml.loss.multiclass.util.MultilabelThresholdPrecisionMatrix;
import com.expleague.ml.loss.multilabel.MultiLabelExactMatch;
import com.expleague.ml.models.MultiClassModel;
import com.expleague.ml.models.multiclass.MCModel;
import com.expleague.ml.models.multilabel.MultiLabelBinarizedModel;
import com.expleague.ml.models.multilabel.MultiLabelModel;

import java.util.function.Function;

/**
 * User: qdeee
 * Date: 04.09.14
 */
@SuppressWarnings("UseOfSystemOutOrSystemErr")
public class ResultsPrinter {
  public static void printResults(final Function computable, final Pool<?> learn, final Pool<?> test, final Func loss, final Func[] metrics) {
    System.out.print("Learn: " + loss.value(DataTools.calcAll(computable, learn.vecData())) + " Test:");
    for (final Func metric : metrics) {
      System.out.print(" " + metric.value(DataTools.calcAll(computable, test.vecData())));
    }
    System.out.println();
  }

  public static void printMulticlassResults(final Function function, final Pool<?> learn, final Pool<?> test) {
    final MCModel mcModel;
    if (function instanceof Ensemble && ((Ensemble) function).last() instanceof FuncJoin) {
      final FuncJoin funcJoin = MCTools.joinBoostingResult((Ensemble) function);
      mcModel = new MultiClassModel(funcJoin);
    } else if (function instanceof MCModel) {
      mcModel = (MCModel) function;
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

  public static void printMultilabelResult(final Function function, final Pool<?> learn, final Pool<?> test) {
    final MultiLabelModel mlModel = MultiLabelTools.extractMultiLabelModel((Trans) function);

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
