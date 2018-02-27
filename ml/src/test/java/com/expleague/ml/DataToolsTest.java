package com.expleague.ml;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.FakePool;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.data.tools.WritableCsvRow;
import com.expleague.ml.meta.items.FakeItem;
import com.expleague.ml.meta.items.QURLItem;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.loss.L2;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.testUtils.TestResourceLoader;
import gnu.trove.map.hash.TObjectIntHashMap;
import junit.framework.TestCase;
import org.junit.Assert;
import org.junit.Test;

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
    TestCase.assertEquals(out.toString(), out1.toString());
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
      TestCase.assertEquals(150, pool.size());
      TestCase.assertEquals(4, pool.features().length);
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
    TestCase.assertEquals(4, split.length);
    System.out.println(Arrays.toString(split));

    final CharSequence[] split1 = CharSeqTools.split("1 2 3", " ");
    TestCase.assertEquals(3, split1.length);
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
    TestCase.assertEquals("0.5\t1:1.0\n0.7\t0:1.0\n", out.toString());
  }

  public void testClassicWrite() throws Exception {
    final StringWriter out = new StringWriter();
    DataTools.writeClassicPoolTo(learn, out);
    final Pool<QURLItem> pool = DataTools.loadFromFeaturesTxt("file", new StringReader(out.toString()));
    TestCase.assertTrue(VecTools.equals(learn.target(L2.class).target, pool.target(L2.class).target));
    TestCase.assertTrue(VecTools.equals(learn.vecData().data(), pool.vecData().data()));
  }

  public void testEmptyCSV() throws Exception {
    DataTools.csvLines(new StringReader("")).forEach(System.out::println);
  }

  public void testQuotesValue() throws Exception {
    TObjectIntHashMap<String> names = new TObjectIntHashMap<>();
    names.put("a", 1);
    names.put("b", 2);
    names.put("c", 3);
    names.put("d", 4);
    WritableCsvRow row = new WritableCsvRow(new CharSeq[4], names);
    row.set("a", 1);
    row.set("b", "Hello \" world");
    row.set("c", "Hello \"\" world 2 \"");
    row.set("d", "Hello \"\" world \"\"\" 3");

    String text = row.names().toString() + "\n" + row.toString();
    DataTools.readCSVWithHeader(new StringReader(text), r -> {
      Assert.assertEquals("Hello \" world", r.asString("b"));
      Assert.assertEquals("Hello \"\" world 2 \"", r.asString("c"));
      Assert.assertEquals("Hello \"\" world \"\"\" 3", r.asString("d"));
    });
  }

  public void testQuotesValue2() throws Exception {
    String header = "userid,deviceid,type,serverts,clientts,relativets,experiments,ostype,country,gender,hardwaremodel,rowsperscreen,columnsperscreen,productid,categoryid,storeid,merchantid,rating,price,position,revenue,search_query,search_categoryid,context_pg,context_catalog,context_search,context_similar,pushtype,pushid,utmparams,split,partition_date";
    String value = "\"1493377914783841726-153-53-581-795312800\",\"1493377914783816035-152-51-581-448137233\",\"productClick\",\"1498368305377\",\"1498368283964\"," +
        "\"4990390594\",\"{productQuestions=enabled, web-payments-enabled=enabled, searchEngine=detectum2, coupon-to-lazy-buyers=coupon-10-7, inviteFriends=enabled, " +
        "productSizeChart=baseline, merchant-offers=enabled, helpshift=enabled, enable-paybox=enabled, rankerPenaltyQuota=baseline, " +
        "groupsInSocialNetworks=groupsInSocialNetworks, catalog-mappings=enabled, productPreviewIndicator=sales, rankerUserModelWeights=purchases_smooth, " +
        "rankerRtUserModels=test30, product-price-elasticy=linear_up_to_5}\",\"android\",\"RU\",\"male\",\"Ixion ML4.5\"\" Ixion ML4.5\"\"\",\"3\",\"2\"," +
        "\"1487567655694365587-198-1-629-1713024225\",\"1473502935938677404-248-2-118-2064455579\",\"1484918498706781429-242-3-26341-1572668229\"," +
        "\"1484917952654400105-31-11-26341-3246049605\",\"4.628158844765343\",\"6.0\",\"100\",\"\",\"\",\"\",\"false\",\"false\",\"false\",\"false\",\"\",\"\",\"\",\"F\"," +
        "\"2017-06-25\"";
    String value1 = "\"1493377914783841726-153-53-581-795312800\",\"1493377914783816035-152-51-581-448137233\",\"productClick\",\"1498368305377\",\"1498368283964\"," +
        "\"4990390594\",\"{productQuestions=enabled, web-payments-enabled=enabled, searchEngine=detectum2, coupon-to-lazy-buyers=coupon-10-7, inviteFriends=enabled, " +
        "productSizeChart=baseline, merchant-offers=enabled, helpshift=enabled, enable-paybox=enabled, rankerPenaltyQuota=baseline, " +
        "groupsInSocialNetworks=groupsInSocialNetworks, catalog-mappings=enabled, productPreviewIndicator=sales, rankerUserModelWeights=purchases_smooth, " +
        "rankerRtUserModels=test30, product-price-elasticy=linear_up_to_5}\",\"android\",\"RU\",\"male\",\"Ixion ML4.5\"\"Ixion ML4.5\"\"\"\",\"3\",\"2\"," +
        "\"1487567655694365587-198-1-629-1713024225\",\"1473502935938677404-248-2-118-2064455579\",\"1484918498706781429-242-3-26341-1572668229\"," +
        "\"1484917952654400105-31-11-26341-3246049605\",\"4.628158844765343\",\"6.0\",\"100\",\"\",\"\",\"\",\"false\",\"false\",\"false\",\"false\",\"\",\"\",\"\",\"F\"," +
        "\"2017-06-25\"";

//    Assert.assertEquals(value, value1);
    DataTools.readCSVWithHeader(new StringReader(header + "\n" + value), r -> {
      Assert.assertEquals(value, r.clone().toString());
    });
  }

  // "1493377914783841726-153-53-581-795312800","1493377914783816035-152-51-581-448137233","productClick","1498367984084","1498367962403","4990069301","{productQuestions=enabled, web-payments-enabled=enabled, searchEngine=detectum2, coupon-to-lazy-buyers=coupon-10-7, inviteFriends=enabled, productSizeChart=baseline, merchant-offers=enabled, helpshift=enabled, enable-paybox=enabled, rankerPenaltyQuota=baseline, groupsInSocialNetworks=groupsInSocialNetworks, catalog-mappings=enabled, productPreviewIndicator=sales, rankerUserModelWeights=purchases_smooth, rankerRtUserModels=test30, product-price-elasticy=linear_up_to_5}","android","RU","male","Ixion ML4.5"" Ixion ML4.5""","3","2","1461934112596872736-249-1-553-47564357","1473502943259066792-27-2-118-1329907027","1460707199853346096-1-3-553-3675721035","1479114284851797703-1-11-118-3521699745","4.647969052224371","2.54","29","","","","false","false","false","false","","","","F","2017-06-25"

  public void testEmptyValue() throws Exception {
    String data =
        "\"\"\"week\",\"categoryId\",\"productOpen\",\"productToCart\",\"productToFavorites\",\"productPurchase\"\n" +
            "\"0\",\"1482944890164121278-235-2-629-2442933182\",\"141414\",\"15308\",\"4180\",\"2184\"\n" +
            "\"2\",\"\",\"0\",\"2\",\"7\",\"0\"\n";

    DataTools.readCSVWithHeader(new StringReader(data), (row) -> {
      Assert.assertNotNull(row.asString("categoryId", ""));
    });
  }

  @Override
  protected boolean isJDK8DependResult() {
    return true;
  }
}
