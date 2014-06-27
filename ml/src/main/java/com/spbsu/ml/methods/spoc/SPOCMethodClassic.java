package com.spbsu.ml.methods.spoc;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.MxIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.IndexTransVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.*;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.func.FuncEnsemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.Optimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MulticlassCodingMatrixModel;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntObjectMap;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class SPOCMethodClassic implements Optimization<LLLogit> {
  protected static final double MX_IGNORE_THRESHOLD = 0.1;
  protected final int k;
  protected final int l;

  protected final double mcStep;
  protected final int mcIters;

  protected final Mx codingMatrix;


  public SPOCMethodClassic(final Mx codingMatrix, final double mcStep, final int mcIters) {
    this.mcStep = mcStep;
    this.mcIters = mcIters;
    this.k = codingMatrix.rows();
    this.l = codingMatrix.columns();
    this.codingMatrix = VecTools.copy(codingMatrix);
    normalizeMx(this.codingMatrix);
  }

  private static void normalizeMx(final Mx codingMatrix) {
    for (MxIterator iter = codingMatrix.nonZeroes(); iter.advance(); ) {
      final double value = iter.value();
      if (Math.abs(value) > MX_IGNORE_THRESHOLD)
        iter.setValue(Math.signum(value));
      else
        iter.setValue(0.0);
    }
  }

  protected Trans createModel(final Func[] binClass, final DataSet learnDS) {
    return new MulticlassCodingMatrixModel(codingMatrix, binClass, MX_IGNORE_THRESHOLD);
  }

  @Override
  public Trans fit(final DataSet learn, final LLLogit llLogit) {
    System.out.println("coding matrix: \n" + codingMatrix.toString());

    final TIntObjectMap<TIntList> indexes = MCTools.splitClassesIdxs(learn);
    final Func[] binClassifiers = new Func[l];
    for (int j = 0; j < l; j++) {
      final TIntList learnIdxs = new TIntLinkedList();
      final TDoubleList target = new TDoubleLinkedList();
      for (int i = 0; i < k; i++) {
        final double code = codingMatrix.get(i, j);
        if (Math.abs(code) > MX_IGNORE_THRESHOLD) {
          final TIntList classIdxs = indexes.get(i);
          target.fill(target.size(), target.size() + classIdxs.size(), Math.signum(code));
          learnIdxs.addAll(classIdxs);
        }
      }

      final DataSet dataSet = new DataSetImpl(
          new VecBasedMx(
              learn.xdim(),
              new IndexTransVec(
                  learn.data(),
                  new RowsPermutation(learnIdxs.toArray(), learn.xdim())
              )
          ),
          new ArrayVec(target.toArray())
      );

      final LLLogit loss = new LLLogit(dataSet.target());
      final BFGrid grid = GridTools.medianGrid(dataSet, 32);
      final GradientBoosting<LLLogit> boosting = new GradientBoosting<LLLogit>(
          new GreedyObliviousTree<L2>(grid, 5),
          mcIters, mcStep
      );
      final ProgressHandler calcer = new ProgressHandler() {
        int index = 0;

        @Override
        public void invoke(Trans partial) {
          if ((index + 1) % 20 == 0) {
            double lvalue = loss.value(partial.transAll(dataSet.data()));
            System.out.print("iter=" + index + ", [learn]LLLogitValue=" + lvalue + "\r");
          }
          index++;
        }
      };
      boosting.addListener(calcer);
      final Ensemble ensemble = boosting.fit(dataSet, loss);
      System.out.println();
      final FuncEnsemble funcEnsemble = new FuncEnsemble(ArrayTools.map(ensemble.models, Func.class, new Computable<Trans, Func>() {
        @Override
        public Func compute(final Trans argument) {
          return (Func)argument;
        }
      }), ensemble.weights);
      binClassifiers[j] = funcEnsemble;
    }
    return createModel(binClassifiers, learn);
  }
}
