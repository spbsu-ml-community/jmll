package com.expleague.ml;

import com.expleague.commons.FileTestCase;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.methods.greedyRegion.GreedyProbLinearRegion;
import com.expleague.ml.testUtils.TestResourceLoader;
import org.junit.Assert;

import java.io.IOException;
import java.util.stream.IntStream;

/**
 * User: solar
 * Date: 12.11.12
 * Time: 16:35
 */
public class GridTest extends FileTestCase {
  public static Pool<?> learn, validate;

  private static synchronized void loadDataSet() {
    try {
      if (learn == null || validate == null) {
        learn = TestResourceLoader.loadPool("features.txt.gz");
        validate = TestResourceLoader.loadPool("featuresTest.txt.gz");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void testFeaturesLoaded() {
    assertEquals(50, learn.vecData().xdim());
    assertEquals(50, validate.vecData().xdim());
    assertEquals(12465, learn.size());
    assertEquals(46596, validate.size());
  }

  public void testGrid1() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
//    assertEquals(624, grid.size());
    checkResultByFile(BFGrid.CONVERTER.convertTo(grid).toString());
  }

  public void testBinary() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    assertEquals(1, grid.row(4).size());
  }

  public void testBinarize1() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    assertEquals(1, grid.row(4).size());
    final byte[] bins = new byte[learn.vecData().xdim()];
    final Vec point = new ArrayVec(learn.vecData().xdim());
    point.set(0, 0.465441);
    point.set(17, 0);

    grid.binarizeTo(point, bins);
    assertEquals(28, bins[0]);
    assertEquals(0, bins[17]);
    assertFalse(grid.bf(28).value(bins));
    assertTrue(grid.bf(27).value(bins));
  }

  public void testBinarize2() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    assertEquals(1, grid.row(4).size());
    final byte[] bins = new byte[learn.vecData().xdim()];
    final Vec point = new ArrayVec(learn.vecData().xdim());
    point.set(0, 0.0);

    grid.binarizeTo(point, bins);
    assertEquals(0, bins[0]);
  }

  public void testBinarize3() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
    assertEquals(1, grid.row(4).size());
    final byte[] bins = new byte[learn.vecData().xdim()];
    final Vec point = new ArrayVec(learn.vecData().xdim());
    point.set(3, 1.0);
    grid.binarizeTo(point, bins);
    final BFGrid.Feature bf = grid.bf(96);

    assertEquals(true, bf.value(bins));
    assertEquals(true, bf.value(point));
  }

  public void testSplitUniform() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(3, grid.size());
    assertEquals(0.1, grid.bf(0).condition());
    assertEquals(0.3, grid.bf(1).condition());
    assertEquals(0.6, grid.bf(2).condition());
  }

  public void testSplitUniformUnsorted() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0.8, 0.5, 0, 0.1, 0.2, 0.3, 0.6, 0.7));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(3, grid.size());
    assertEquals(0.1, grid.bf(0).condition());
    assertEquals(0.3, grid.bf(1).condition());
    assertEquals(0.6, grid.bf(2).condition());
  }

  public void testSplitBinary() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0, 1, 1, 1, 1, 1, 1));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(1, grid.size());
    assertEquals(0., grid.bf(0).condition());
  }

  public void testSplitBinaryIncorrect1() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(1, 1, 1, 1, 1, 1, 1, 1));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(0, grid.size());
  }

  public void testSplitBinaryIncorrect2() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 1, 1, 1, 1, 1, 1, 1));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(1, grid.size());
  }

  public void testBinarize4() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8));
    final VecDataSet ds = new VecDataSetImpl(data, null);
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    final byte[] bin = new byte[1];
    grid.binarizeTo(new ArrayVec(0.), bin);
    assertEquals(0, bin[0]);
    grid.binarizeTo(new ArrayVec(1.), bin);
    assertEquals(3, bin[0]);
  }

  public void testSameFeatures() {
      final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0.0, 0.5, 0.3));
      final VecDataSet ds = new VecDataSetImpl(data, null);
      final BFGrid grid = GridTools.medianGrid(ds, 32);
      assertEquals(1, grid.size());
  }

  public void testIterativeP_cxVsFull() {
    final double lambda = 0.00001;
    final int n = 1000;
    final int c = 300;
    double D_prev = Double.NEGATIVE_INFINITY;
    double S_prev = Double.NEGATIVE_INFINITY;

    for (int k = 0; k < n; k++) {
      double D = 0;
      for (int i = c; i < n; i++) {
        D += Math.exp(-lambda * Math.abs(i - k));
      }
      double S = 0;
      for (int i = 0; i < n; i++) {
        S += Math.exp(-lambda * Math.abs(i - k));
      }
      if (Double.isFinite(D_prev)) {
        Assert.assertEquals(D, D_prev + Math.exp(-lambda * Math.abs(c - k)) - Math.exp(-lambda * (n - k - 1)), 0.01);
        Assert.assertEquals(S,S_prev + Math.exp(-lambda * k) - Math.exp(-lambda * (n - k - 1)), 0.01);
      }
      D_prev = D;
      S_prev = S;
    }
  }

  public void testProbLinearScoreFromLambda() {
    final VecDataSet ds = learn.vecData();
    final BFGrid grid = GridTools.medianGrid(ds, 32);
    final L2 target = learn.target(L2.class);
    WeightedLoss<L2> wloss = new WeightedLoss<>(target, IntStream.range(0, target.dim()).map(idx -> 1).toArray());
    BinarizedDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    GreedyProbLinearRegion.ScoreFromLambda scoreFromLambda = new GreedyProbLinearRegion.ScoreFromLambda(bds, IntStream.range(0, learn.size()).toArray(), wloss, true, grid.bf(10));

    Vec x = new ArrayVec(10, 10);
    Vec grad = new ArrayVec(2);
    scoreFromLambda.gradientTo(x, grad);
    Vec realGrad = new ArrayVec(2);
    double v = scoreFromLambda.value(x);
    x.adjust(0, MathTools.EPSILON);
    realGrad.set(0, (v - scoreFromLambda.value(x)) / MathTools.EPSILON);
    x.adjust(0, -MathTools.EPSILON);
    x.adjust(1, MathTools.EPSILON);
    realGrad.set(1, (v - scoreFromLambda.value(x)) / MathTools.EPSILON);
    x.adjust(1, -MathTools.EPSILON);
    assertTrue(VecTools.distanceL2(grad, realGrad) < 1e-3);
  }

  public void testPlotProbLinearScoreFromLambda() {
    final VecDataSet ds = learn.vecData();
    final BFGrid grid = GridTools.medianGrid(ds, 32);
    final L2 target = learn.target(L2.class);
    WeightedLoss<L2> wloss = new WeightedLoss<>(target, IntStream.range(0, target.dim()).map(idx -> 1).toArray());
    BinarizedDataSet bds = ds.cache().cache(Binarize.class, VecDataSet.class).binarize(grid);

    GreedyProbLinearRegion.ScoreFromLambda scoreFromLambda = new GreedyProbLinearRegion.ScoreFromLambda(bds, IntStream.range(0, learn.size()).toArray(), wloss, true, grid.bf(10));

    Vec x = new ArrayVec(10, 10);
    for (int i = 0; i < 100; i++) {
      x.set(0, i * 0.1 - 5);
      System.out.println(x + "\t" + scoreFromLambda.value(x));
    }
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    loadDataSet();
  }

  @Override
  protected String getInputFileExtension() {
    return ".txt";
  }

  @Override
  protected String getResultFileExtension() {
    return ".txt";
  }

  @Override
  protected String getTestDataPath() {
    try {
      return TestResourceLoader.getFullPath("grid/");
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
  }
}
