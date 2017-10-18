package com.expleague.ml.cli.output.printers;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.models.multiclass.MCModel;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.multiclass.util.ConfusionMatrix;

import static com.expleague.commons.math.vectors.VecTools.scale;

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
    if (partial instanceof Ensemble) {
      final Ensemble ensemble = (Ensemble) partial;
      final double step = ensemble.wlast();
      final Trans model = ensemble.last();

      //caching boosting results
      VecTools.append(learnValues, VecTools.scale(model.transAll(learn.data()), step));
      VecTools.append(testValues, VecTools.scale(model.transAll(test.data()), step));
    }

    iteration++;
    if (iteration % itersForOut == 0) {
      final IntSeq learnPredicted;
      final IntSeq testPredicted;

      if (partial instanceof Ensemble) {
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
