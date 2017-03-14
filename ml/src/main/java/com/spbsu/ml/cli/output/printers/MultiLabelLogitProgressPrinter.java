package com.spbsu.ml.cli.output.printers;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.Func;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.blockwise.BlockwiseMultiLabelLogit;
import com.spbsu.ml.loss.multilabel.MultiLabelExactMatch;
import com.spbsu.ml.loss.multilabel.MultiLabelHammingLoss;
import com.spbsu.ml.loss.multilabel.MultiLabelMacroFScore;
import com.spbsu.ml.loss.multilabel.MultiLabelMicroFScore;
import com.spbsu.ml.models.multiclass.MCModel;
import com.spbsu.ml.models.multilabel.MultiLabelModel;

import java.util.ArrayList;
import java.util.List;

import static com.spbsu.commons.math.vectors.VecTools.append;
import static com.spbsu.commons.math.vectors.VecTools.scale;

/**
 * User: qdeee
 * Date: 03.04.15
 */
public class MultiLabelLogitProgressPrinter implements ProgressHandler {
  private final VecDataSet learn;
  private final VecDataSet test;

  private final BlockwiseMultiLabelLogit learnLogit;
  private final BlockwiseMultiLabelLogit testLogit;
  private final Mx learnValues;
  private final Mx testValues;

  private final List<Func> learnMetrics = new ArrayList<>();
  private final List<Func> testMetrics = new ArrayList<>();

  private final int itersForOut;
  private int iteration = 0;

  public MultiLabelLogitProgressPrinter(final Pool<?> learn, final Pool<?> test) {
    this(learn, test, 10);
  }

  public MultiLabelLogitProgressPrinter(final Pool<?> learn, final Pool<?> test, final int itersForOut) {
    this.learn = learn.vecData();
    this.test = test.vecData();

    this.learnLogit = learn.target(BlockwiseMultiLabelLogit.class);
    this.testLogit = test.target(BlockwiseMultiLabelLogit.class);
    this.learnValues = new VecBasedMx(learn.size(), learnLogit.blockSize());
    this.testValues = new VecBasedMx(test.size(), testLogit.blockSize());

    this.learnMetrics.add(learn.target(MultiLabelExactMatch.class));
    this.learnMetrics.add(learn.target(MultiLabelMicroFScore.class));
    this.learnMetrics.add(learn.target(MultiLabelMacroFScore.class));
    this.learnMetrics.add(learn.target(MultiLabelHammingLoss.class));

    this.testMetrics.add(test.target(MultiLabelExactMatch.class));
    this.testMetrics.add(test.target(MultiLabelMicroFScore.class));
    this.testMetrics.add(test.target(MultiLabelMacroFScore.class));
    this.testMetrics.add(test.target(MultiLabelHammingLoss.class));

    this.itersForOut = itersForOut;
  }

  @Override
  public void invoke(final Trans partial) {
    if (isBoostingProcess(partial)) {
      final Ensemble ensemble = (Ensemble) partial;
      final double step = ensemble.wlast();
      final FuncJoin model = (FuncJoin) ensemble.last();

      //caching boosting results
      append(learnValues, scale(model.transAll(learn.data()), step));
      append(testValues, scale(model.transAll(test.data()), step));
    }

    iteration++;
    if (iteration % itersForOut == 0) {
      final Mx learnPredicted;
      final Mx testPredicted;

      if (isBoostingProcess(partial)) {
        learnPredicted = VecTools.toBinary(VecTools.copy(learnValues));
        testPredicted = VecTools.toBinary(VecTools.copy(testValues));

      } else if (partial instanceof MCModel) {
        final MultiLabelModel mcModel = (MultiLabelModel) partial;
        learnPredicted = mcModel.predictLabelsAll(learn.data());
        testPredicted = mcModel.predictLabelsAll(test.data());
      } else return;

      System.out.print(iteration);
      System.out.print(" " + learnLogit.value(learnValues));
      System.out.print(" " + testLogit.value(testValues));
      System.out.print(" { ");
      for (Func learnMetric : learnMetrics) {
        System.out.print(learnMetric.value(learnPredicted));
        System.out.print(" ");
      }
      System.out.printf("} {");
      for (Func testMetric : testMetrics) {
        System.out.print(testMetric.value(testPredicted));
        System.out.print(" ");
      }
      System.out.printf("}\n");
    }
  }

  private static boolean isBoostingProcess(final Trans partial) {
    return partial instanceof Ensemble && ((Ensemble) partial).last() instanceof FuncJoin;
  }
}
