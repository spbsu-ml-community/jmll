package com.spbsu.ml.methods.multiclass.spoc;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.idxtrans.RowsPermutation;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.IndexTransVec;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.MulticlassCodingMatrixModel;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.list.linked.TIntLinkedList;
import gnu.trove.map.TIntObjectMap;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class SPOCMethodClassic extends VecOptimization.Stub<BlockwiseMLLLogit> {
  protected static final double MX_IGNORE_THRESHOLD = 0.1;

  protected final VecOptimization<LLLogit> weak;
  protected final Mx codeMatrix;

  public SPOCMethodClassic(final Mx codeMatrix, final VecOptimization<LLLogit> weak) {
    this.weak = weak;
    this.codeMatrix = VecTools.copy(codeMatrix);
    CMLHelper.normalizeMx(this.codeMatrix, MX_IGNORE_THRESHOLD);
  }

  protected Trans createModel(final Func[] binClass, final VecDataSet learnDS, final BlockwiseMLLLogit llLogit) {
    return new MulticlassCodingMatrixModel(codeMatrix, binClass, MX_IGNORE_THRESHOLD);
  }

  @Override
  public Trans fit(final VecDataSet learn, final BlockwiseMLLLogit llLogit) {
//    System.out.println("coding matrix: \n" + codeMatrix.toString());

    final TIntObjectMap<TIntList> indexes = MCTools.splitClassesIdxs(llLogit.labels());
    final int k = codeMatrix.rows();
    final int l = codeMatrix.columns();
    final Func[] binClassifiers = new Func[l];
    for (int j = 0; j < l; j++) {
      final TIntList learnIdxs = new TIntLinkedList();
      final TDoubleList target = new TDoubleLinkedList();
      for (int i = 0; i < k; i++) {
        final double code = codeMatrix.get(i, j);
        if (Math.abs(code) > MX_IGNORE_THRESHOLD) {
          final TIntList classIdxs = indexes.get(i);
          target.fill(target.size(), target.size() + classIdxs.size(), Math.signum(code));
          learnIdxs.addAll(classIdxs);
        }
      }

      final VecDataSet dataSet = new VecDataSetImpl(
          new VecBasedMx(
              learn.xdim(),
              new IndexTransVec(
                  learn.data(),
                  new RowsPermutation(learnIdxs.toArray(), learn.xdim())
              )
          ), learn);
      final LLLogit loss = new LLLogit(new ArrayVec(target.toArray()), learn);
      binClassifiers[j] = (Func) weak.fit(dataSet, loss);
    }
    return createModel(binClassifiers, learn, llLogit);
  }
}
