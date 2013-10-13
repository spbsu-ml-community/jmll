package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.Histogram;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.DataSetImpl;

/**
 * User: solar
 * Date: 03.12.12
 * Time: 20:28
 */
public class DataToolsTest extends GridTest {
  public void testBootStrap() {
    final DataSet bootstrap = DataTools.bootstrap(learn, new FastRandom(0));
    final Vec row = bootstrap.data().row(10);
//    System.out.println(row.toString());
    assertTrue(learn.data().row(7656).equals(row));
    assertEquals(learn.target().get(7656), bootstrap.target().get(10));
  }

  public void testBuildHistogram() {
    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8));
    DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
    final BFGrid grid = GridTools.medianGrid(ds, 3);
    assertEquals(3, grid.size());
    BinarizedDataSet bds = new BinarizedDataSet(ds, grid);
    final Histogram histogram = bds.buildHistogram(new ArrayVec(0, 0, 0, 0, 1, 0, 0, 1), new ArrayVec(8), ArrayTools.sequence(0, 8));
    final double[] weights = new double[grid.size()];
    final double[] sums = new double[grid.size()];
    final double[] scores = new double[grid.size()];
    histogram.score(scores, new Histogram.Judge() {
      int index = 0;
      @Override
      public double score(double sum, double sum2, double weight, int bf) {
        weights[index] = weight;
        sums[index] = sum;
        index++;
        return 0;
      }
    });
    assertEquals(2., sums[0]);
    assertEquals(6., weights[0]);
    assertEquals(2., sums[1]);
    assertEquals(4., weights[1]);
    assertEquals(1., sums[2]);
    assertEquals(2., weights[2]);
  }
}
