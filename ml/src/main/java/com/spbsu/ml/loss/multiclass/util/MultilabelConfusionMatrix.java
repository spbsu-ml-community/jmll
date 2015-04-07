package com.spbsu.ml.loss.multiclass.util;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.table.TableBuilder;

/**
 * User: qdeee
 * Date: 24.03.15
 */
public class MultilabelConfusionMatrix {
  private ConfusionMatrix[] matrixes;

  public MultilabelConfusionMatrix(final Mx targets, final Mx predicted) {
    matrixes = new ConfusionMatrix[targets.columns()];
    for (int j = 0; j < matrixes.length; j++) {
      matrixes[j] = new ConfusionMatrix(
          VecTools.toIntSeq(targets.col(j)),
          VecTools.toIntSeq(predicted.col(j))
      );
    }
  }

  public String toSummaryString() {
    final double[] microPrecision = new double[matrixes.length];
    final double[] macroPrecision = new double[matrixes.length];
    final double[] macroRecall = new double[matrixes.length];
    final double[] macroFScore = new double[matrixes.length];

    for (int i = 0; i < matrixes.length; i++) {
      microPrecision[i] = matrixes[i].getMicroPrecision();
      macroPrecision[i] = matrixes[i].getMacroPrecision();
      macroRecall[i] = matrixes[i].getMacroRecall();
      macroFScore[i] = matrixes[i].getMacroF1Measure();
    }

    final TableBuilder tableBuilder = new TableBuilder();
    tableBuilder.setHeader("Metric\\Label", ArrayTools.sequence(0, matrixes.length));
//    tableBuilder.addRow("Micro precision: ",  microPrecision);
//    tableBuilder.addRow("Micro recall: ",     microPrecision);
//    tableBuilder.addRow("Micro F1-measure: ", microPrecision);
    tableBuilder.addRow("Macro precision: ",  macroPrecision);
    tableBuilder.addRow("Macro recall: ",     macroRecall);
    tableBuilder.addRow("Macro F1-measure: ", macroFScore);
    return "=== Summary ===\n" + tableBuilder.build();
  }

  public String toClassDetailsString() {
    final double[] precision0 = new double[matrixes.length];
    final double[] precision1 = new double[matrixes.length];
    final double[] recall0 = new double[matrixes.length];
    final double[] recall1 = new double[matrixes.length];
    final double[] fScore0 = new double[matrixes.length];
    final double[] fScore1 = new double[matrixes.length];

    for (int i = 0; i < matrixes.length; i++) {
      precision0[i] = matrixes[i].getPrecision(0);
      precision1[i] = matrixes[i].getPrecision(1);
      recall0[i] = matrixes[i].getRecall(0);
      recall1[i] = matrixes[i].getRecall(1);
      fScore0[i] = matrixes[i].getF1Measure(0);
      fScore1[i] = matrixes[i].getF1Measure(1);
    }

    final TableBuilder tableBuilder = new TableBuilder();
    final String table = tableBuilder
        .setHeader("metric\\label", ArrayTools.sequence(0, matrixes.length))
        .addRow("[0] precision", precision0)
        .addRow("[0] recall",    recall0)
        .addRow("[0] f1-score",  fScore0)
        .addRow("[1] precision", precision1)
        .addRow("[1] recall",    recall1)
        .addRow("[1] f1-score",  fScore1)
        .build();
    return "=== Detailed Accuracy By Class ===\n" + table;
  }
}
