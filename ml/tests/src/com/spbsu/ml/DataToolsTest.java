package com.spbsu.ml;

import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.ml.data.DSIterator;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.impl.DataSetImpl;

/**
 * User: solar
 * Date: 03.12.12
 * Time: 20:28
 */
public class DataToolsTest extends GridTest {
  public void testBuildHistogram() {
//    final VecBasedMx data = new VecBasedMx(1, new ArrayVec(0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8));
//    DataSet ds = new DataSetImpl(data, new ArrayVec(data.rows()));
//    final BFGrid grid = GridTools.medianGrid(ds, 3);
//    assertEquals(3, grid.size());
//    BinarizedDataSet bds = new BinarizedDataSet(ds, grid);
//    final MSEHistogram result = new MSEHistogram(grid, new ArrayVec(0, 0, 0, 0, 1, 0, 0, 1), new ArrayVec(8));
//
//    bds.aggregate(result, ArrayTools.sequence(0, 8));
//    final MSEHistogram histogram = result;
//    final double[] weights = new double[grid.size()];
//    final double[] sums = new double[grid.size()];
//    final double[] scores = new double[grid.size()];
//    histogram.score(scores, new MSEHistogram.Judge() {
//      int index = 0;
//      @Override
//      public double score(double sum, double sum2, double weight, int bf) {
//        weights[index] = weight;
//        sums[index] = sum;
//        index++;
//        return 0;
//      }
//    });
//    assertEquals(2., sums[0]);
//    assertEquals(6., weights[0]);
//    assertEquals(2., sums[1]);
//    assertEquals(4., weights[1]);
//    assertEquals(1., sums[2]);
//    assertEquals(2., weights[2]);
  }

  public void testExtendDataset() throws Exception {
    DataSet ds = new DataSetImpl(
        new VecBasedMx(2,
            new ArrayVec(1, 2,
                         3, 4)
        ),
        new ArrayVec(0.1,
                     0.2)
    );
    final DataSet extDs = DataTools.extendDataset(ds, new ArrayVec(5., 6.), new ArrayVec(7., 8.));
    for (DSIterator iter = extDs.iterator(); iter.advance(); ) {
      System.out.println(iter.y() + "\t" + iter.x().toString());
    }
  }
}
