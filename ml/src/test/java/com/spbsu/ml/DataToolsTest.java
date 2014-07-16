package com.spbsu.ml;

import java.io.StringReader;
import java.io.StringWriter;


import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.meta.DSItem;

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

  public void testDSSave() throws Exception {
    final StringWriter out = new StringWriter();
    DataTools.writePoolTo(learn, out);
    checkResultByFile(out.getBuffer());
  }

  public void testDSSaveLoad() throws Exception {
    final StringWriter out = new StringWriter();
    DataTools.writePoolTo(learn, out);
    final Pool<? extends DSItem> pool = DataTools.readPoolFrom(new StringReader(out.toString()));
    StringWriter out1 = new StringWriter();
    DataTools.writePoolTo(pool, out1);
    assertEquals(out.toString(), out1.toString());
  }

  public void testExtendDataset() throws Exception {
    final ArrayVec target = new ArrayVec(0.1,
        0.2);
    VecDataSet ds = new VecDataSetImpl(
        new VecBasedMx(2,
            new ArrayVec(1, 2,
                         3, 4)
        ),
        null
    );
    final VecDataSet extDs = DataTools.extendDataset(ds, new ArrayVec(5., 6.), new ArrayVec(7., 8.));

    for (int i = 0; i < extDs.length(); i++) {
      System.out.println(target.get(i) + "\t" + extDs.at(i).toString());
    }
  }
}
