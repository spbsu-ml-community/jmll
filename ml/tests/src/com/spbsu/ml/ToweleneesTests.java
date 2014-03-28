package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.GreedyPolynomialExponentRegion;
import com.spbsu.ml.methods.trees.GreedyContinuesObliviousSoftBondariesRegressionTree;
import com.spbsu.ml.methods.trees.GreedyExponentialObliviousTree;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.ContinousObliviousTree;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.models.PolynomialExponentRegion;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * Created by towelenee on 28.03.14.
 */
//My tests for my experements
public class ToweleneesTests extends MethodsTests {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public void testPCAOTBoost() {
    DataSet mas[] = new DataSet[2];
    doPCA(mas);
    DataSet myValidate = mas[1], myLearn = mas[0];
    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(myLearn, 32), 6), rng), 2000, 0.02);
    new addBoostingListeners<SatL2>(boosting, new SatL2(myLearn.target()), myLearn, myValidate);
  }

  public void testCOTBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
      new BootstrapOptimization(new GreedyContinuesObliviousSoftBondariesRegressionTree(rng, learn, GridTools.medianGrid(learn, 32), 6, 10, true, 1, 0, 0, 1e5), rng), 2000, 0.01);
    new addBoostingListeners<SatL2>(boosting, new SatL2(learn.target()), learn, validate);
  }

  public void testEOTBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
      new BootstrapOptimization<L2>(
        new GreedyExponentialObliviousTree(
          GridTools.medianGrid(learn, 32), 6, 2500), rng), 2000, 0.04);
    new addBoostingListeners<SatL2>(boosting, new SatL2(learn.target()), learn, validate);
  }

  public void testContinousObliviousTree() {
    ScoreCalcer scoreCalcerValidate = new ScoreCalcer(/*" On validate data Set loss = "*/"\t", validate);
    ScoreCalcer scoreCalcerLearn = new ScoreCalcer(/*"On learn data Set loss = "*/"\t", learn);
    for (int depth = 1; depth <= 6; depth++) {
      ContinousObliviousTree tree = new GreedyContinuesObliviousSoftBondariesRegressionTree(
        rng,
        learn,
        GridTools.medianGrid(learn, 32), depth, 10, true, 1, 0.1, 1, 1e5).fit(learn, new L2(learn.target()));
      //for(int i = 0; i < 10/*learn.target().ydim()*/;i++)
      // System.out.println(learn.target().get(i) + "= " + tree.value(learn.data().row(i)));
      System.out.print("Oblivious Tree deapth = " + depth);
      scoreCalcerLearn.invoke(tree);
      scoreCalcerValidate.invoke(tree);

      System.out.println();
      //System.out.println(tree.toString());
    }
  }

  public void testJB() {
    for (int j = 0; j < learn.data().columns(); j++) {
      double mean = 0;
      double mc2 = 0, mc3 = 0, mc4 = 0;
      for (int i = 0; i < learn.power(); i++)
        mean += learn.data().get(i, j);
      mean /= learn.power();
      for (int i = 0; i < learn.power(); i++) {
        mc2 += Math.pow(learn.data().get(i, j) - mean, 2);
        mc3 += Math.pow(learn.data().get(i, j) - mean, 3);
        mc4 += Math.pow(learn.data().get(i, j) - mean, 4);
      }
      if (mc2 != 0) {
        mc2 /= learn.power();
        mc3 /= learn.power();
        mc4 /= learn.power();
        System.out.println(mean);
        System.out.println(mc2);
        System.out.println(mc3);
        System.out.println(mc4);
        double K = mc4 / Math.pow(mc2, 2);
        double S = mc3 / Math.pow(mc2, 1.5);
        System.out.println(j + "= " + learn.power() * (S * S + 0.25 * Math.pow(K - 3, 2)) / 6.0);
      }
    }
  }

  public void testExponentialObliviousTree() {
    ScoreCalcer scoreCalcerValidate = new ScoreCalcer(/*" On validate data Set loss = "*/"\t", validate);
    ScoreCalcer scoreCalcerLearn = new ScoreCalcer(/*"On learn data Set loss = "*/"\t", learn);
    for (int depth = 1; depth <= 6; depth++) {
      ContinousObliviousTree tree = new GreedyExponentialObliviousTree(
        GridTools.medianGrid(learn, 32), depth, 15).fit(learn, new WeightedLoss<L2>(new L2(learn.target())));
      //for(int i = 0; i < 10/*learn.target().ydim()*/;i++)
      // System.out.println(learn.target().get(i) + "= " + tree.value(learn.data().row(i)));
      System.out.print("Oblivious Tree deapth = " + depth);
      scoreCalcerLearn.invoke(tree);
      scoreCalcerValidate.invoke(tree);

      System.out.println();
      //System.out.println(tree.toString());
    }
  }

  //Not safe can make diffrent size for learn and test
  public Mx cutNonContinuesFeatures(Mx ds, boolean continues[]) {

    int continuesFeatures = 0;
    for (int j = 0; j < ds.columns(); j++)
      for (int i = 0; i < ds.rows(); i++)
        if ((Math.abs(ds.get(i, j)) > 1e-7) && (Math.abs(ds.get(i, j) - 1) > 1e-7)) {
          continues[j] = true;
          continuesFeatures++;
          break;
        }
    int reg[] = new int[ds.columns()];
    int cnt = 0;
    for (int i = 0; i < ds.columns(); i++)
      if (continues[i])
        reg[i] = cnt++;
    Mx data = new VecBasedMx(ds.rows(), continuesFeatures);
    for (int i = 0; i < ds.rows(); i++)
      for (int j = 0; j < ds.columns(); j++)
        if (continues[j])
          data.set(i, reg[j], ds.get(i, j));
    return data;
  }

  public boolean checkEigenDecomposion(Mx mx, Mx q, Mx sigma) {
    for (int i = 0; i < mx.rows(); i++) {
      for (int j = 0; j < mx.columns(); j++)
        if (Math.abs(VecTools.multiply(mx, q.row(i)).get(j) - q.get(i, j) * sigma.get(i, i)) > 1e-9) {
          System.out.println(VecTools.multiply(mx, q.row(i)).get(j));
          System.out.println(q.get(i, j) * sigma.get(i, i));
          return false;
        }
    }
    return true;
  }

  public Mx LQInverse(Mx mx) {
    if (mx.rows() != mx.columns())
      throw new IllegalArgumentException("Matrix must be square");
    Mx l = new VecBasedMx(mx.rows(), mx.columns());
    Mx q = new VecBasedMx(mx.rows(), mx.columns());
    Mx mxCopy = new VecBasedMx(mx);

    VecTools.householderLQ(mx, l, q);

    Mx lq = VecTools.multiply(l, q);
    for (int i = 0; i < mx.rows(); i++)
      for (int j = 0; j < mx.rows(); j++)
        //if(Math.abs(mxCopy.get(i,j) - lq.get(i,j)) > 1e-3)
        System.out.println(mxCopy.get(i, j) - lq.get(i, j));
    Mx ans = VecTools.multiply(VecTools.transpose(q), VecTools.inverseLTriangle(l));
    System.out.println("1 = " + VecTools.multiply(mx, l));
    System.exit(0);
    return ans;
  }

  Mx calculateCoovMatrix(Mx data) {
    Mx res = new VecBasedMx(data.columns(), data.columns());
    double mean[] = new double[data.columns()];
    for (int j = 0; j < data.columns(); j++) {
      for (int i = 0; i < data.rows(); i++)
        mean[j] += data.get(i, j);
      mean[j] /= data.rows();
    }
    for (int i = 0; i < data.columns(); i++)
      for (int j = 0; j < data.columns(); j++) {
        double cov = 0;
        for (int k = 0; k < data.rows(); k++)
          cov += (data.get(k, i) - mean[i]) * (data.get(k, j) - mean[j]);
        res.set(i, j, cov);
      }
    return res;
  }

  public void doPCA(DataSet[] mas) {
    boolean continues[] = new boolean[learn.xdim()];
    Mx learnMx = cutNonContinuesFeatures(learn.data(), continues);
    Mx validateMx = cutNonContinuesFeatures(validate.data(), continues);
    Mx mx = calculateCoovMatrix(learnMx);
    Mx q = new VecBasedMx(mx.columns(), mx.rows());
    Mx sigma = new VecBasedMx(mx.columns(), mx.rows());
    VecTools.eigenDecomposition(mx, q, sigma);
    System.out.println(mx);
    //assertTrue(checkEigeDecomposion(mx, q, sigma));
    //System.exit(0);
    //q = LQInverse(q);
    for (int i = 0; i < learnMx.rows(); i++) {
      Vec nw = GreedyPolynomialExponentRegion.solveLinearEquationUsingLQ(q, learnMx.row(i));
      for (int j = 0; j < learnMx.columns(); j++)
        learnMx.set(i, j, nw.get(j));
    }
    for (int i = 0; i < validateMx.rows(); i++) {
      Vec nw = GreedyPolynomialExponentRegion.solveLinearEquationUsingLQ(q, validateMx.row(i));
      for (int j = 0; j < validateMx.columns(); j++)
        validateMx.set(i, j, nw.get(j));
    }
    //Normalization
    for (int i = 0; i < learnMx.columns(); i++) {
      double max = -1e10, mn = 1e10;
      for (int j = 0; j < learnMx.rows(); j++) {
        max = Math.max(max, learnMx.get(j, i));
        mn = Math.min(mn, learnMx.get(j, i));
      }
      for (int j = 0; j < validateMx.rows(); j++) {
        max = Math.max(max, validateMx.get(j, i));
        mn = Math.min(mn, validateMx.get(j, i));
      }
      for (int j = 0; j < learnMx.rows(); j++) {
        learnMx.set(j, i, (learnMx.get(j, i) - mn) / (max - mn));
      }
      for (int j = 0; j < validateMx.rows(); j++) {
        validateMx.set(j, i, (validateMx.get(j, i) - mn) / (max - mn));
      }
    }

    int reg[] = new int[learn.xdim()], cnt = learnMx.columns(), cntCont = 0;
    //continues = new boolean[learn.xdim()];
    for (int i = 0; i < learn.xdim(); i++)
      if (!continues[i])
        reg[i] = cnt++;
      else
        reg[i] = cntCont++;
    Mx learnOut = new VecBasedMx(learn.power(), learn.xdim());
    Mx validateOut = new VecBasedMx(validate.power(), validate.xdim());
    for (int i = 0; i < learn.power(); i++)
      for (int j = 0; j < learn.xdim(); j++)
        if (!continues[j])
          learnOut.set(i, reg[j], learn.data().get(i, j));
        else
          learnOut.set(i, reg[j], learnMx.get(i, reg[j]));
    for (int i = 0; i < validate.power(); i++) {
      for (int j = 0; j < validate.xdim(); j++)
        if (!continues[j])
          validateOut.set(i, reg[j], validate.data().get(i, j));
        else
          validateOut.set(i, reg[j], validateMx.get(i, reg[j]));
    }
    mas[0] = new DataSetImpl(learnOut, learn.target());
    mas[1] = new DataSetImpl(validateOut, validate.target());
    //mas[0] = learn;
    //mas[1] = validate;


  }

  public void testPSAContinousObliviousTree() {
    DataSet mas[] = new DataSet[2];
    doPCA(mas);
    DataSet myValidate = mas[1], myLearn = mas[0];
    System.out.println(myLearn.data().row(0));
    System.out.println(learn.data().row(0));
    ScoreCalcer scoreCalcerValidate = new ScoreCalcer(/*" On validate data Set loss = "*/"\t", myValidate);
    ScoreCalcer scoreCalcerLearn = new ScoreCalcer(/*"On learn data Set loss = "*/"\t", myLearn);
    //System.out.println(learn.data());
    for (int depth = 1; depth <= 6; depth++) {
      ContinousObliviousTree tree = new GreedyContinuesObliviousSoftBondariesRegressionTree(rng, myLearn, GridTools.medianGrid(myLearn, 32), depth, 1, true, 1, 0.1, 1, 1e5).fit(myLearn, new L2(myLearn.target()));
      //for(int i = 0; i < 10/*learn.target().ydim()*/;i++)
      // System.out.println(learn.target().get(i) + "= " + tree.value(learn.data().row(i)));
      System.out.print("Oblivious Tree deapth = " + depth);
      scoreCalcerLearn.invoke(tree);
      scoreCalcerValidate.invoke(tree);

      System.out.println();
      //System.out.println(tree.toString());
    }
  }

  public void testPSACOTboost() {
    DataSet mas[] = new DataSet[2];
    doPCA(mas);
    DataSet myValidate = mas[1], myLearn = mas[0];


    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
      new GreedyContinuesObliviousSoftBondariesRegressionTree(rng, myLearn, GridTools.medianGrid(myLearn, 32), 6, 6, true, 1, 0.1, 1, 1e6),
      2000, 0.1);
    new addBoostingListeners<SatL2>(boosting, new SatL2(myLearn.target()), myLearn, myValidate);
  }

  public void testDebugContinousObliviousTree() {
    //ScoreCalcer scoreCalcerValidate = new ScoreCalcer(" On validate data Set loss = ", validate);
    double[] data = {0, 1, 2};
    double[] target = {0, 1, 2};

    DataSet debug = new DataSetImpl(data, target);
    ScoreCalcer scoreCalcerLearn = new ScoreCalcer(" On learn data Set loss = ", debug);
    for (int depth = 1; depth <= 1; depth++) {
      ContinousObliviousTree tree = new GreedyContinuesObliviousSoftBondariesRegressionTree(rng, debug, GridTools.medianGrid(debug, 32), depth, 1, true, 10, 0.1, 1, 1e5).fit(debug, new L2(debug.target()));
      System.out.print("Oblivious Tree deapth = " + depth);
      scoreCalcerLearn.invoke(tree);
      System.out.println(tree.toString());
      System.out.println();
    }
  }

  public void testObliviousTreeFail() throws FileNotFoundException {
    int depth = 6;
    Scanner scanner = new Scanner(new File("./ml/tests/data/badMx.txt"));
    Vec vec = new ArrayVec(learn.power());
    System.out.println(learn.power());
    for (int i = 0; i < learn.power(); i++)
      vec.set(i, Double.parseDouble(scanner.next()));
    ObliviousTree tree = (ObliviousTree) new GreedyObliviousTree(GridTools.medianGrid(learn, 32), depth).fit(learn, new L2(vec));
    System.out.println(tree);
  }

  public void testGreedyPolynomialExponentRegion() {
    PolynomialExponentRegion region = new GreedyPolynomialExponentRegion(GridTools.medianGrid(learn, 32), 0, 0).fit(learn, new WeightedLoss<L2>(new L2(learn.target())));
    ScoreCalcer scoreCalcerLearn = new ScoreCalcer(" On learn data Set loss = ", learn);
    ScoreCalcer scoreCalcerValidate = new ScoreCalcer(" On learn data Set loss = ", validate);
    scoreCalcerLearn.invoke(region);
    scoreCalcerValidate.invoke(region);
  }

  public void testGreedyPolynomialExponentRegionBoost() {
    GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(new BootstrapOptimization<L2>(new GreedyPolynomialExponentRegion(GridTools.medianGrid(learn, 32), 1, 2500), rng), 5000, 0.05);
    new addBoostingListeners<SatL2>(boosting, new SatL2(learn.target()), learn, validate);
  }

}
