package com.spbsu.ml;

import com.spbsu.commons.FileTestCase;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.impl.DataSetImpl;

import java.io.IOException;

/**
 * User: solar
 * Date: 12.11.12
 * Time: 16:35
 */
public class GridTest extends FileTestCase {
  public static DataSet learn, validate;

  static private synchronized void loadDataSet() {
    try {
      if (learn == null || validate == null) {
        learn = DataTools.loadFromFeaturesTxt("./ml/tests/data/features.txt.gz");
        validate = DataTools.loadFromFeaturesTxt("./ml/tests/data/featuresTest.txt.gz");
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void testFeaturesLoaded() {
    assertEquals(50, learn.xdim());
    assertEquals(50, validate.xdim());
    assertEquals(12465, learn.power());
    assertEquals(46596, validate.power());
  }

  public void testGrid1() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn, 32);
//    assertEquals(624, grid.size());
    checkResultByFile(BFGrid.CONVERTER.convertTo(grid).toString());
  }

  public void testBinary() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn, 32);
    assertEquals(1, grid.row(4).size());
  }

  public void testBinarize1() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn, 32);
    assertEquals(1, grid.row(4).size());
    byte[] bins = new byte[learn.xdim()];
    Vec point = new ArrayVec(learn.xdim());
    point.set(0, 0.465441);
    point.set(17, 0);

    grid.binarize(point, bins);
    assertEquals(28, bins[0]);
    assertEquals(0, bins[17]);
    assertFalse(grid.bf(28).value(bins));
    assertTrue(grid.bf(27).value(bins));
  }

  public void testBinarize2() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn, 32);
    assertEquals(1, grid.row(4).size());
    byte[] bins = new byte[learn.xdim()];
    Vec point = new ArrayVec(learn.xdim());
    point.set(0, 0.0);

    grid.binarize(point, bins);
    assertEquals(0, bins[0]);
  }

  public void testBinarize3() throws IOException {
    final BFGrid grid = GridTools.medianGrid(learn, 32);
    assertEquals(1, grid.row(4).size());
    byte[] bins = new byte[learn.xdim()];
    Vec point = new ArrayVec(learn.xdim());
    point.set(3, 1.0);
    grid.binarize(point, bins);
    final BFGrid.BinaryFeature bf = grid.bf(96);

    assertEquals(true, bf.value(bins));
    assertEquals(true, bf.value(point));
  }

  public void testSplitUniform() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8));
    DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(3, grid.size());
    assertEquals(0.1, grid.bf(0).condition);
    assertEquals(0.3, grid.bf(1).condition);
    assertEquals(0.6, grid.bf(2).condition);
  }

  public void testSplitUniformUnsorted() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0.8, 0.5, 0, 0.1, 0.2, 0.3, 0.6, 0.7));
    DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(3, grid.size());
    assertEquals(0.1, grid.bf(0).condition);
    assertEquals(0.3, grid.bf(1).condition);
    assertEquals(0.6, grid.bf(2).condition);
  }

  public void testSplitBinary() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0, 1, 1, 1, 1, 1, 1));
    DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(1, grid.size());
    assertEquals(0., grid.bf(0).condition);
  }

  public void testSplitBinaryIncorrect1() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(1, 1, 1, 1, 1, 1, 1, 1));
    DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(0, grid.size());
  }

  public void testSplitBinaryIncorrect2() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 1, 1, 1, 1, 1, 1, 1));
    DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(1, grid.size());
  }

  public void testBinarize4() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8));
    DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    byte[] bin = new byte[1];
    grid.binarize(new ArrayVec(0.), bin);
    assertEquals(0, bin[0]);
    grid.binarize(new ArrayVec(1.), bin);
    assertEquals(3, bin[0]);
  }

  public void testSameFeatures() {
      final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0.0, 0.5, 0.3));
      DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
      final BFGrid grid = GridTools.medianGrid(ds, 32);
      assertEquals(1, grid.size());
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
    return "./ml/tests/data/grid/";
  }
}
