package com.spbsu.ml.cli.output.printers;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.ml.ProgressHandler;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncJoin;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.loss.multiclass.util.ConfusionMatrix;
import com.spbsu.ml.models.multiclass.MCModel;

import static com.spbsu.commons.math.vectors.VecTools.append;
import static com.spbsu.commons.math.vectors.VecTools.scale;

/**
 * User: qdeee
 * Date: 04.09.14
 */
public class MulticlassProgressPrinter implements ProgressHandler {
  private final VecDataSet learn;
  private final VecDataSet test;

  private final BlockwiseMLLLogit learnMLLLogit;
  private final BlockwiseMLLLogit testMLLLogit;
  private final Mx learnValues;
  private final Mx testValues;

  private final int itersForOut;
  int iteration = 0;

  public MulticlassProgressPrinter(final Pool<?> learn, final Pool<?> test) {
    this(learn, test, 10);
  }

  public MulticlassProgressPrinter(final Pool<?> learn, final Pool<?> test, final int itersForOut) {
    this.learn = learn.vecData();
    this.test = test.vecData();

    this.learnMLLLogit = learn.target(BlockwiseMLLLogit.class);
    this.testMLLLogit = test.target(BlockwiseMLLLogit.class);
    assert learnMLLLogit.classesCount() == testMLLLogit.classesCount();

    this.learnValues = new VecBasedMx(learn.size(), learnMLLLogit.classesCount() - 1);
    this.testValues = new VecBasedMx(test.size(), testMLLLogit.classesCount() - 1);
    this.itersForOut = itersForOut;
  }

  @Override
  public void invoke(final Trans partial) {
    if (isBoostingMulticlassProcess(partial)) {
      final Ensemble ensemble = (Ensemble) partial;
      final double step = ensemble.wlast();
      final FuncJoin model = (FuncJoin) ensemble.last();

      //caching boosting results
      append(learnValues, scale(model.transAll(learn.data()), step));
      append(testValues, scale(model.transAll(test.data()), step));
    }

    iteration++;
    if (iteration % itersForOut == 0) {
      final IntSeq learnPredicted;
      final IntSeq testPredicted;

      if (isBoostingMulticlassProcess(partial)) {
        learnPredicted = convertTransResults(learnValues);
        testPredicted = convertTransResults(testValues);
      } else if (partial instanceof MCModel) {
        final MCModel mcModel = (MCModel) partial;
        learnPredicted = VecTools.toIntSeq(mcModel.bestClassAll(learn.data()));
        testPredicted = VecTools.toIntSeq(mcModel.bestClassAll(test.data()));
      } else return;

      System.out.print(iteration);

      System.out.print(" " + learnMLLLogit.value(learnValues));
      System.out.print(" " + testMLLLogit.value(testValues));

      final ConfusionMatrix learnConfusionMatrix = new ConfusionMatrix(learnMLLLogit.labels(), learnPredicted);
      System.out.print("\t" + learnConfusionMatrix.oneLineReport());

      final ConfusionMatrix testConfusionMatrix = new ConfusionMatrix(testMLLLogit.labels(), testPredicted);
      System.out.print("\t" + testConfusionMatrix.oneLineReport());
      System.out.println();
    }
  }

  private static boolean isBoostingMulticlassProcess(final Trans partial) {
    return partial instanceof Ensemble && ((Ensemble) partial).last() instanceof FuncJoin;
  }

  private static IntSeq convertTransResults(final Mx trans) {
    final int[] result = new int[trans.rows()];
    for (int i = 0; i < trans.rows(); i++) {
      final Vec row = trans.row(i);
      final int bestClass = VecTools.argmax(row);
      result[i] = row.get(bestClass) > 0 ? bestClass : row.dim();
    }
    return new IntSeq(result);
  }
}
