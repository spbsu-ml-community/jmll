package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.HierTools;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.multiclass.MCMacroPrecision;
import com.spbsu.ml.loss.multiclass.MCMicroPrecision;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.list.linked.TDoubleLinkedList;
import junit.framework.TestCase;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class CodingMatrixLearningTest extends TestCase {
  protected Mx B;
  protected CodingMatrixLearning method;
  protected DataSet learn;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    B = new VecBasedMx(
        2,
        new ArrayVec(
            1,  0,
            1, -1,
            -1,  1,
            0,  1)
    );
    method = new CodingMatrixLearning(B.rows(), B.columns());

    final TDoubleLinkedList borders = new TDoubleLinkedList();
    borders.addAll(new double[] {0.038125, 0.07625, 0.114375, 0.1525});
    learn = HierTools.loadRegressionAsMC("./ml/tests/data/features.txt.gz", B.rows(), borders);
  }

  public void testFindMatrix() throws Exception {
    final Mx S = method.createSimilarityMatrix(learn, DataTools.splitClassesIdxs(learn));
    final int iters = 50;
    final double step = 0.5;
    final double lambdaC = 0.2;
    final double lambdaR = 0.3;
    final double lambda1 = 0.4;
    final Mx matrixB = method.findMatrixB(S, iters, step, lambdaC, lambdaR, lambda1);
    System.out.println(matrixB.toString());
  }

  public void testCreateConstraintsMatrix() throws Exception {

    final Mx constraintsMatrix = CodingMatrixLearning.createConstraintsMatrix(B);
    System.out.println(constraintsMatrix.toString());
  }

  public void testSimilarityMatrix() throws Exception {
    final Mx similarityMatrix = method.createSimilarityMatrix(learn, DataTools.splitClassesIdxs(learn));
    System.out.println(similarityMatrix.toString());
  }

  public void testFit() throws Exception {
    final MultiClassModel model = (MultiClassModel) method.fit(learn, new LLLogit(learn.target()));
    final Vec predict = model.bestClassAll(learn.data());
    final Func microPrecision = new MCMicroPrecision(learn.target());
    final Func macroPrecision = new MCMacroPrecision(learn.target());
    System.out.println("microP = " + microPrecision.value(predict));
    System.out.println("macroP = " + macroPrecision.value(predict));
  }
}
