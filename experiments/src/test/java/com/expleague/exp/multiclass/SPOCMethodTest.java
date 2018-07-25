package com.expleague.exp.multiclass;

import com.expleague.exp.multiclass.weak.CustomWeakBinClass;
import com.expleague.exp.multiclass.weak.CustomWeakMultiClass;
import com.expleague.commons.io.StreamTools;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.MCTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.TargetMeta;
import com.expleague.ml.meta.impl.fake.FakeTargetMeta;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multiclass.spoc.AbstractCodingMatrixLearning;
import com.expleague.ml.methods.multiclass.spoc.SPOCMethodClassic;
import com.expleague.ml.methods.multiclass.spoc.impl.CodingMatrixLearning;
import com.expleague.ml.methods.multiclass.spoc.impl.CodingMatrixLearningGreedy;
import com.expleague.ml.methods.multiclass.spoc.impl.CodingMatrixLearningGreedyParallels;
import com.expleague.ml.models.multiclass.MCModel;
import com.expleague.ml.models.multiclass.MulticlassCodingMatrixModel;
import com.expleague.ml.testUtils.TestResourceLoader;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import junit.framework.TestCase;

import java.io.IOException;

/**
* User: qdeee
* Date: 07.05.14
*/
public class SPOCMethodTest extends TestCase {
  private static final double[] hierBorders = new double[] {0.038125, 0.07625, 0.114375, 0.1525, 0.61};
  private static final double[] classicBorders = new double[]{0.06999, 0.13999, 0.40999, 0.60999, 0.61};

  protected Pool<?> learn;
  protected Pool<?> test;
  protected Mx S;

  protected int k;
  protected int l;

  private synchronized void initDefaultData() throws IOException {
    if (learn == null || test == null) {
      final TDoubleList borders = new TDoubleArrayList(hierBorders);
      learn = TestResourceLoader.loadPool("features.txt.gz");
      test = TestResourceLoader.loadPool("featuresTest.txt.gz");
      final IntSeq learnTarget = MCTools.transformRegressionToMC(learn.target(L2.class).target, borders.size(), borders);
      final IntSeq testTarget = MCTools.transformRegressionToMC(test.target(L2.class).target, borders.size(), borders);
      learn.addTarget(TargetMeta.create("spoc", "", FeatureMeta.ValueType.INTS), learnTarget);
      test.addTarget(TargetMeta.create("spoc", "", FeatureMeta.ValueType.INTS), testTarget);

//      final CharSequence mxStr = StreamTools.readStream(TestResourceLoader.loadResourceAsStream("multiclass/regression_based/features.txt.similarityMx"));
      final CharSequence mxStr = StreamTools.readStream(TestResourceLoader.loadResourceAsStream("multiclass/regression_based/features-simmatrix-classic.txt"));
      S = MathTools.CONVERSION.convert(mxStr, Mx.class);

      k = borders.size();
      l = 5;
    }
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    initDefaultData();
  }

  private void printResult(final MCModel model) {
    System.out.println(MCTools.evalModel(model, learn, "[LEARN]", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST]", false));
    System.out.println(MCTools.evalModel(model, learn, getName(), true) + MCTools.evalModel(model, test, "", true));
  }

  private void fitModel(final AbstractCodingMatrixLearning matrixLearning, final int iters, final double step) {
    final Mx codingMatrix = matrixLearning.trainCodingMatrix(S);
//      if (!CodingMatrixLearning.checkConstraints(codeMatrix)) {
//        throw new IllegalStateException("Result matrix is out of constraints");
//      }
    final VecOptimization method = new SPOCMethodClassic(codingMatrix, new CustomWeakBinClass(iters, step));
    final MulticlassCodingMatrixModel model = (MulticlassCodingMatrixModel) method.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    printResult(model);
  }

  public void testBaseline() throws Exception {
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(100, 0.5);
    final MCModel model = (MCModel) customWeakMultiClass.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    printResult(model);
  }

  public void testMathFit() throws Exception {
    fitModel(new CodingMatrixLearning(k, l, 3.0, 2.5, 7.0, 1.8),
        100, 0.3);
  }

  public void testGreedyFit() throws Exception {
    fitModel(new CodingMatrixLearningGreedy(k, l, 3.0, 2.5, 7.0),
        200, 0.3);
  }

  public void testParallelsGreedyFit() throws Exception {
    fitModel(new CodingMatrixLearningGreedyParallels(k, l, 3.0, 2.5, 7.0),
        200, 0.3);
  }

  public void _testBaselineBigDS() throws Exception {
    final Pool<?> pool = TestResourceLoader.loadPool("multiclass/ds_letter/letter.tsv.gz");
    pool.addTarget(TargetMeta.create("letter", "", FeatureMeta.ValueType.INTS),
        VecTools.toIntSeq(pool.target(L2.class).target));

    final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100501), 0.8, 0.2);
    final Pool<?> learn = pool.sub(idxs[0]);
    final Pool<?> test = pool.sub(idxs[1]);

    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(300, 0.7);
    final MCModel model = (MCModel) customWeakMultiClass.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    System.out.println(MCTools.evalModel(model, learn, "[LEARN]", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST]", false));
    System.out.println(MCTools.evalModel(model, learn, getName(), true) + MCTools.evalModel(model, test, "", true));
  }

  public void _testMathFitBigDS() throws Exception {
    final Pool<?> pool = TestResourceLoader.loadPool("multiclass/ds_letter/letter.tsv.gz");
    pool.addTarget(TargetMeta.create("letter", "", FeatureMeta.ValueType.INTS), VecTools.toIntSeq(pool.target(L2.class).target));

    final int[][] idxs = DataTools.splitAtRandom(pool.size(), new FastRandom(100500), 0.8, 0.2);
    final Pool<?> learn = pool.sub(idxs[0]);
    final Pool<?> test = pool.sub(idxs[1]);

    final CharSequence mxStr = StreamTools.readStream(TestResourceLoader.loadResourceAsStream("multiclass/ds_letter/letter.similarityMx"));
    final Mx similarityMx = MathTools.CONVERSION.convert(mxStr, Mx.class);


    final CodingMatrixLearning codingMatrixLearning = new CodingMatrixLearning(26, 10, 10.0, 2.5, 5.0, 1.8);
    final Mx codeMx = codingMatrixLearning.findMatrixB(similarityMx);

    final SPOCMethodClassic spoc = new SPOCMethodClassic(codeMx, new CustomWeakBinClass(300, 0.7));
    final MCModel model = (MCModel) spoc.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    System.out.println(MCTools.evalModel(model, learn, "[LEARN]", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST]", false));
    System.out.println(MCTools.evalModel(model, learn, getName(), true) + MCTools.evalModel(model, test, "", true));
  }
}