package com.spbsu.exp.multiclass;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.exp.multiclass.weak.CustomWeakBinClass;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.SubPool;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.FakeTargetMeta;
import com.spbsu.ml.methods.spoc.ECOCCombo;
import com.spbsu.ml.models.MulticlassCodingMatrixModel;
import com.spbsu.ml.testUtils.TestResourceLoader;
import junit.framework.TestCase;

import java.io.FileInputStream;
import java.io.IOException;

public class ECOCComboTest extends TestCase {
  private static Pool<?> learn;
  private static Pool<?> test;
  private static Mx S;

  private synchronized static void init() throws IOException {
    if (learn == null || test == null) {
      final Pool<?> pool = TestResourceLoader.loadPool("multiclass/catalog.tsv");
      pool.addTarget(new FakeTargetMeta(pool.vecData(), FeatureMeta.ValueType.INTS),
                     VecTools.toIntSeq(pool.target(L2.class).target)
      );
      final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100500), 0.8, 0.5);
      learn = new SubPool<>(pool, idxs[0]);
      test = new SubPool<>(pool, idxs[1]);

      final CharSequence mxStr = StreamTools.readStream(new FileInputStream("/Users/qdeee/datasets/catalog-final/catalog50-1stlevel-gt5000.tsv.simmx"));
      S = MathTools.CONVERSION.convert(mxStr, Mx.class);

    }
  }

  @Override
  protected void setUp() throws Exception {
    init();
  }

  public void testFit() throws Exception {
    final BlockwiseMLLLogit mllLogit = learn.target(BlockwiseMLLLogit.class);
    final VecDataSet vecDataSet = learn.vecData();

    final int k = MCTools.countClasses(mllLogit.labels());
    final ECOCCombo ecocComboMethod = new ECOCCombo(k, k, 5.0, 2.5, 3.0, S, new CustomWeakBinClass(10, 1.5));
    final Action<MulticlassCodingMatrixModel> listener = new Action<MulticlassCodingMatrixModel>() {
      @Override
      public void invoke(final MulticlassCodingMatrixModel model) {
        System.out.println("L == " + model.getInternalModel().ydim());
        System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", true));
        System.out.println(MCTools.evalModel(model, test, "[TEST] ", true));
      }
    };
    ecocComboMethod.addListener(listener);
    final MulticlassCodingMatrixModel model = (MulticlassCodingMatrixModel) ecocComboMethod.fit(vecDataSet, mllLogit);
    System.out.println("\n\n\n");
    System.out.println(MCTools.evalModel(model, learn, "[LEARN] ", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST] ", false));

  }
}