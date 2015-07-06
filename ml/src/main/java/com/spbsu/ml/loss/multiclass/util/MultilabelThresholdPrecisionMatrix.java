package com.spbsu.ml.loss.multiclass.util;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.table.TableBuilder;

/**
 * Created by irlab on 25.06.2015.
 */
public class MultilabelThresholdPrecisionMatrix {
  private final Mx scores;
  private final Mx targets;
  private final int probThresholdBuckets;
  private final String name;

  public MultilabelThresholdPrecisionMatrix(final Mx scores, final Mx targets, final int probThresholdBuckets, final String name) {
    this.scores = scores;
    this.targets = targets;
    this.probThresholdBuckets = probThresholdBuckets;
    this.name = name;
  }

  public String toThresholdPrecisionMatrix() {
    final TableBuilder tableBuilder = new TableBuilder();
    final String[] header = new String[targets.columns() * 2];
    for (int i = 0; i < targets.columns(); i++) {
      header[i * 2] = "class " + i + " precision";
      header[i * 2 + 1] = "class " + i + " recall";
    }
    tableBuilder.setHeader("threshold", header);

    for (int thresholdNum = 0; thresholdNum <= probThresholdBuckets; thresholdNum++) {
      final double threshold = ((double)thresholdNum) / probThresholdBuckets;
      final double[] tableRow = new double[targets.columns() * 2];
      for (int classId = 0; classId < targets.columns(); classId++) {
        int cntf = 0;
        int cntr = 0;
        int cntfr = 0;
        for (int example = 0; example < targets.rows(); example++) {
          final double prob = 1./ (1. + Math.exp(-scores.get(example, classId)));
          final boolean isFound = targets.get(example, classId) > 0;
          if (prob >= threshold) {
            cntf ++;
            if (isFound)
              cntfr ++;
          }
          if (isFound)
            cntr++;
        }
        final double precision = cntfr / (cntf + 1E-12);
        final double recall = cntfr / (cntr + 1E-12);
        tableRow[classId * 2] = precision;
        tableRow[classId * 2 + 1] = recall;
      }
      tableBuilder.addRow("" + threshold, tableRow);
    }

    final String table = tableBuilder.build();
    return name + "\n" + table;
  }
}
