package com.spbsu.ml.methods;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.*;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.HierTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.MLLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.multiclass.MCMacroPrecision;
import com.spbsu.ml.loss.multiclass.MCMicroPrecision;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.MultiClassModel;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.linked.TDoubleLinkedList;
import gnu.trove.map.TIntObjectMap;
import junit.framework.TestCase;

import java.io.IOException;
import java.util.Random;

/**
 * User: qdeee
 * Date: 07.05.14
 */
public class CodingMatrixLearningTest extends TestCase {
//  protected CodingMatrixLearning method;
  protected DataSet learn;
  protected DataSet test;
  protected int k;
  protected int l;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    final TDoubleLinkedList borders = new TDoubleLinkedList();
    borders.addAll(new double[] {0.038125, 0.07625, 0.114375, 0.1525, 0.61});
    learn = HierTools.loadRegressionAsMC("./ml/tests/data/features.txt.gz", borders.size(), borders);
    test = HierTools.loadRegressionAsMC("./ml/tests/data/featuresTest.txt.gz", borders.size(), borders);
    k = borders.size();
    l = k;
  }

  public void testFindMatrix() throws Exception {
    final CodingMatrixLearning method = new CodingMatrixLearning(k, l);
    final Mx S = CodingMatrixLearning.createSimilarityMatrix(learn, DataTools.splitClassesIdxs(learn));
    final Mx matrixB = method.findMatrixB(S, 0.75);
    System.out.println(matrixB.toString());
  }

  public static void testCreateConstraintsMatrix() throws Exception {
    final Mx B = new VecBasedMx(
        2,
        new ArrayVec(
            1, -1,
            -1,  1,
            0,  1)
    );
    final int k = B.rows();
    final int l = B.columns();
    final Vec b = new ArrayVec(2* k * l + 2* l + k);
    {
      for (int i = 0; i < 2*k*l; i++)
        b.set(i, 1.);
      for (int i = 2* k * l; i < 2*k*l + 2*l; i++)
        b.set(i, -2.);
      for (int i = 2* k * l + 2* l; i < 2*k*l + 2*l + k; i++)
        b.set(i, -1.);
    }
    System.out.println("vector b:");
    System.out.println(b.toString());

    final Mx constraintsMatrix = CodingMatrixLearning.createConstraintsMatrix(B);
    System.out.println("constraints matrix:");
    System.out.println(constraintsMatrix.toString());
  }

  public void testSimilarityMatrix() throws Exception {
    final Mx similarityMatrix = CodingMatrixLearning.createSimilarityMatrix(learn);
    System.out.println(similarityMatrix.toString());
  }

  public void testBaseline() {
    final MLLLogit learnLoss = new MLLLogit(learn.target());
    final MLLLogit testLoss = new MLLLogit(test.target());
    final BFGrid grid = GridTools.medianGrid(learn, 32);
    final GradientBoosting<MLLLogit> boosting = new GradientBoosting<MLLLogit>(new MultiClass(new GreedyObliviousTree(grid, 5),
        new Computable<Vec, L2>() {
          @Override
          public L2 compute(Vec argument) {
            return new SatL2(argument);
          }
        }
    ), 1000, 0.5);
    final ProgressHandler calcer = new ProgressHandler() {
      int index = 0;

      @Override
      public void invoke(Trans partial) {
        if ((index + 1) % 20 == 0) {
          double lvalue = learnLoss.value(partial.transAll(learn.data()));
          double tvalue = testLoss.value(partial.transAll(test.data()));
          System.out.println("iter=" + index + ", [learn]MLLLogitValue=" + lvalue + ", [test]MLLLogitValue=" + tvalue);
        }
        index++;
      }
    };
    boosting.addListener(calcer);
    Ensemble ensemble = boosting.fit(learn, learnLoss);
    MultiClassModel model = HierarchicalClassification.joinBoostingResults(ensemble);
    final Vec learnPredict = model.bestClassAll(learn.data());
    final Vec testPredict = model.bestClassAll(test.data());

    final Func.Stub learnP = new MCMicroPrecision(learn.target());
    final Func.Stub testP = new MCMicroPrecision(test.target());
    double learnPVal = learnP.value(learnPredict);
    double testPVal = testP.value(testPredict);
    System.out.println("learnPVal = " + learnPVal);
    System.out.println("testPVal = " + testPVal);
  }

  public void testFit() throws Exception {
    final CodingMatrixLearning method = new CodingMatrixLearning(k, l);
    final MultiClassModel model = (MultiClassModel) method.fit(learn, new LLLogit(learn.target()));
    {
      final Vec predict = model.bestClassAll(learn.data());
      final Func microPrecision = new MCMicroPrecision(learn.target());
      final Func macroPrecision = new MCMacroPrecision(learn.target());
      System.out.println("[LEARN]microP = " + microPrecision.value(predict));
      System.out.println("[LEARN]macroP = " + macroPrecision.value(predict));
    }
    {
      final Vec predict = model.bestClassAll(test.data());
      final Func microPrecision = new MCMicroPrecision(test.target());
      final Func macroPrecision = new MCMacroPrecision(test.target());
      System.out.println("[TEST]microP = " + microPrecision.value(predict));
      System.out.println("[TEST]macroP = " + macroPrecision.value(predict));
    }
  }

  public void testHandjob() throws IOException {
    final Mx initB = new VecBasedMx(
        2,
        new ArrayVec(
            1, -1,
            -1,  1,
            0,  1)
    );
    final int k = initB.rows();
    CodingMatrixLearning method = new CodingMatrixLearning(initB);
    System.out.println("initB\n" + initB);
    System.out.println("constraints\n" + CodingMatrixLearning.createConstraintsMatrix(initB));

    DataSet ds = HierTools.loadRegressionAsMC("./ml/tests/data/features.txt.gz", k, new TDoubleLinkedList());
    final Mx S = CodingMatrixLearning.createSimilarityMatrix(ds, DataTools.splitClassesIdxs(ds));
    System.out.println("S:\n" + S.toString());
    final Mx matrixB = method.findMatrixB(S, 0.75);
  }
}
