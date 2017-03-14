package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.FakePool;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.items.FakeItem;
import com.spbsu.ml.meta.items.QURLItem;
import com.spbsu.ml.testUtils.TestResourceLoader;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.Arrays;

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
    final StringWriter out1 = new StringWriter();
    DataTools.writePoolTo(pool, out1);
    assertEquals(out.toString(), out1.toString());
  }

  public void testExtendDataset() throws Exception {
    final ArrayVec target = new ArrayVec(0.1,
        0.2);
    final VecDataSet ds = new VecDataSetImpl(
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

  public void testLibSvmRead() throws Exception {
    try (InputStream stream = TestResourceLoader.loadResourceAsStream("multiclass/iris.libfm")) {
      final Pool<FakeItem> pool = DataTools.loadFromLibSvmFormat(new InputStreamReader(stream));
      assertEquals(150, pool.size());
      assertEquals(4, pool.features().length);
    }
  }

  public void testSparse() throws Exception {
    final SparseVec sparseVec = new SparseVec(0);
    System.out.println(sparseVec.dim());
    sparseVec.set(4, 50.);
    System.out.println(sparseVec.dim());
    System.out.println(sparseVec.get(4));
    System.out.println(sparseVec.get(3));

  }

  public void testSplit() throws Exception {
    final CharSequence[] split = CharSeqTools.split("1 2 3 ", ' ');
    assertEquals(4, split.length);
    System.out.println(Arrays.toString(split));

    final CharSequence[] split1 = CharSeqTools.split("1 2 3", " ");
    assertEquals(3, split1.length);
    System.out.println(Arrays.toString(split1));
  }

  public void testLibfmWrite() throws Exception {
    final Mx data = new VecBasedMx(2, new ArrayVec(
        0.0, 1.0,
        1.0, 0.0
    ));
    final Vec target = new ArrayVec(0.5, 0.7);
    final FakePool pool = new FakePool(data, target);
    final StringWriter out = new StringWriter();
    DataTools.writePoolInLibfmFormat(pool, out);
    assertEquals("0.5\t1:1.0\n0.7\t0:1.0\n", out.toString());
  }

  public void testClassicWrite() throws Exception {
    final StringWriter out = new StringWriter();
    DataTools.writeClassicPoolTo(learn, out);
    final Pool<QURLItem> pool = DataTools.loadFromFeaturesTxt("file", new StringReader(out.toString()));
    assertTrue(VecTools.equals(learn.target(L2.class).target, pool.target(L2.class).target));
    assertTrue(VecTools.equals(learn.vecData().data(), pool.vecData().data()));
  }

  @Override
  protected boolean isJDK8DependResult() {
    return true;
  }
}
