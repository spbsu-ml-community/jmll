package com.spbsu.exp;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.tree.IntTree;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.HierTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.FakeTargetMeta;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.MultiClass;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.hierarchical.HierarchicalClassification;
import com.spbsu.ml.methods.hierarchical.HierarchicalRefinedClassification;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MCModel;
import com.spbsu.ml.models.MultiClassModel;
import com.spbsu.ml.testUtils.FakePool;
import com.spbsu.ml.testUtils.TestResourceLoader;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import junit.framework.TestCase;

/**
 * User: qdeee
 * Date: 28.07.14
 */
public class HierClassTests extends TestCase {
  private static Pool<?> learn;
  private static Pool<?> test;
  private static IntTree tree;

  private static int iters;
  private static double step;

  private synchronized void init() throws Exception {
    if (learn == null || test == null) {
      learn = TestResourceLoader.loadPool("features.txt.gz");
      test = TestResourceLoader.loadPool("featuresTest.txt.gz");

      final TDoubleList borders = new TDoubleArrayList();
      final IntSeq learnTarget = MCTools.transformRegressionToMC(learn.target(L2.class).target, 16, borders);
      final IntSeq testTarget = MCTools.transformRegressionToMC(test.target(L2.class).target, borders.size(), borders);

      final HierTools.TreeBuilder treeBuilder = new HierTools.TreeBuilder(450);
      treeBuilder.createFromOrderedMulticlass(learnTarget);

      tree = treeBuilder.releaseTree();
      final TIntIntMap map = treeBuilder.releaseMapping();

      learn.addTarget(new FakeTargetMeta(learn.vecData(), FeatureMeta.ValueType.INTS), MCTools.mapTarget(learnTarget, map));
      test.addTarget(new FakeTargetMeta(test.vecData(), FeatureMeta.ValueType.INTS), MCTools.mapTarget(testTarget, map));

      iters = 200;
      step = 1.5;
    }
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    init();
  }

  public void testBaseline() throws Exception {
    final TIntIntMap labelsMap = new TIntIntHashMap();
    learn.addTarget(new FakeTargetMeta(learn.vecData(), FeatureMeta.ValueType.INTS), MCTools.normalizeTarget(learn.target(BlockwiseMLLLogit.class).labels(), labelsMap));
    test.addTarget(new FakeTargetMeta(test.vecData(), FeatureMeta.ValueType.INTS), MCTools.mapTarget(test.target(BlockwiseMLLLogit.class).labels(), labelsMap));
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(iters, step);
    final MCModel model = (MCModel) customWeakMultiClass.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    System.out.println(MCTools.evalModel(model, learn, "[LEARN]", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST]", false));
  }

  public void testHierClass() throws Exception {
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(iters, step);
    final HierarchicalClassification hierarchicalClassification = new HierarchicalClassification(customWeakMultiClass, tree);
    final HierarchicalModel model = (HierarchicalModel) hierarchicalClassification.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    System.out.println(MCTools.evalModel(model, learn, "[LEARN]", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST]", false));
  }

  public void testHierRefinedClass() throws Exception {
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(iters, step);
    final CustomWeakBinClass customWeakBinClass = new CustomWeakBinClass(iters, step);
    final HierarchicalRefinedClassification hierarchicalRefinedClassification = new HierarchicalRefinedClassification(customWeakBinClass, customWeakMultiClass, tree);
    final HierarchicalModel model = (HierarchicalModel) hierarchicalRefinedClassification.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    System.out.println(MCTools.evalModel(model, learn, "[LEARN]", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST]", false));
  }

  private static class CustomWeakMultiClass extends VecOptimization.Stub<BlockwiseMLLLogit> {
    private final int iters;
    private final double step;

    public CustomWeakMultiClass(int iters, double step) {
      this.iters = iters;
      this.step = step;
    }

    @Override
    public Trans fit(final VecDataSet learnData, final BlockwiseMLLLogit loss) {
      final BFGrid grid = GridTools.medianGrid(learnData, 32);
      final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), SatL2.class), iters, step);

      final IntSeq intTarget = ((BlockwiseMLLLogit) loss).labels();
      final FakePool ds = new FakePool(learnData.data(), intTarget);

      System.out.println(prepareComment(intTarget));
      final ProgressHandler calcer = new ProgressHandler() {
        int iter = 0;

        @Override
        public void invoke(Trans partial) {
          if ((iter + 1) % 20 == 0) {
            if (((Ensemble) partial).last() instanceof MultiClassModel) {
              final MultiClassModel model = MCTools.joinBoostingResults((Ensemble) partial);
              final Mx x = model.transAll(learnData.data());
              double value = loss.value(x);
              System.out.println("iter=" + iter + ", [learn]MLLLogitValue=" + String.format("%.10f", value) + ", stats=" + MCTools.evalModel(model, ds, "[LEARN]", true) + "\r");
            }
          }
          iter++;
        }
      };
      boosting.addListener(calcer);
      final Ensemble ensemble = boosting.fit(learnData, loss);
      final MCModel model = MCTools.joinBoostingResults(ensemble);
      System.out.println("\n\n");
      return model;
    }

    private static String prepareComment(final IntSeq labels) {
      final StringBuilder builder = new StringBuilder("Class entries count: { ");
      final int countClasses = MCTools.countClasses(labels);
      for (int i = 0; i < countClasses; i++) {
        builder.append(i)
            .append(" : ")
            .append(MCTools.classEntriesCount(labels, i))
            .append(", ");
      }
      return builder.append("}").toString();
    }
  }

  private static class CustomWeakBinClass extends VecOptimization.Stub<LLLogit> {
    private final int iters;
    private final double step;

    private CustomWeakBinClass(final int iters, final double step) {
      this.iters = iters;
      this.step = step;
    }

    @Override
    public Trans fit(final VecDataSet learn, final LLLogit targetFunc) {
      final Vec binClassTarget = targetFunc.labels();
      final IntSeq intBinClassTarget = VecTools.toIntSeq(binClassTarget);
      final IntSeq mcTarget = MCTools.normalizeTarget(intBinClassTarget, new TIntIntHashMap());

      final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(iters, step);
      final MultiClassModel mcm = (MultiClassModel) customWeakMultiClass.fit(learn, new BlockwiseMLLLogit(mcTarget, learn));
      return mcm.getInternModel().dirs()[0];
    }
  }

}
