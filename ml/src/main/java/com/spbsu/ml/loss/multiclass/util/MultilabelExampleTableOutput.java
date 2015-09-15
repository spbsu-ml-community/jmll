package com.spbsu.ml.loss.multiclass.util;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.ml.data.tools.Pool;

/**
 * Created by irlab on 25.06.2015.
 */
public class MultilabelExampleTableOutput {
  private final Mx scores;
  private final Mx targets;
  private final Pool<?> pool;
  private final String name;

  public MultilabelExampleTableOutput(final Mx scores, final Mx targets, final Pool<?> pool, final String name) {
    this.scores = scores;
    this.targets = targets;
    this.pool = pool;
    this.name = name;
  }

  public String toExampleTableMatrix() {
    final StringBuilder out = new StringBuilder();
    out.append(name);

    out.append("example");
    for (int i = 0; i < targets.columns(); i++)
      out.append("\tclass ").append(i).append(" score");
    for (int i = 0; i < targets.columns(); i++)
      out.append("\tclass ").append(i).append(" target");
    out.append('\n');

    for (int example = 0; example < targets.rows(); example++) {
      out.append(pool.data().at(example).id());
      final double[] tableRow = new double[targets.columns() * 2];
      for (int classId = 0; classId < targets.columns(); classId++) {
        final double score = 1. / (1. + Math.exp(-scores.get(example, classId)));
        out.append('\t').append(score);
      }
      for (int classId = 0; classId < targets.columns(); classId++) {
        out.append('\t').append(targets.get(example, classId));
      }
      out.append('\n');
    }

    return out.toString();
  }
}
