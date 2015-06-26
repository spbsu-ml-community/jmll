package com.spbsu.exp.multiclass;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.tree.IntTree;
import com.spbsu.exp.multiclass.weak.CustomWeakBinClass;
import com.spbsu.exp.multiclass.weak.CustomWeakMultiClass;
import com.spbsu.ml.data.tools.HierTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.impl.fake.FakeTargetMeta;
import com.spbsu.ml.methods.multiclass.hierarchical.HierarchicalClassification;
import com.spbsu.ml.methods.multiclass.hierarchical.HierarchicalRefinedClassification;
import com.spbsu.ml.models.multiclass.HierarchicalModel;
import com.spbsu.ml.models.multiclass.MCModel;
import com.spbsu.ml.testUtils.TestResourceLoader;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntIntMap;
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

  private void printResult(final MCModel model) {
    System.out.println(MCTools.evalModel(model, learn, "[LEARN]", false));
    System.out.println(MCTools.evalModel(model, test, "[TEST]", false));
    System.out.println(MCTools.evalModel(model, learn, getName(), true) + MCTools.evalModel(model, test, "", true));
  }

  public void testHierClass() throws Exception {
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(iters, step);
    final HierarchicalClassification hierarchicalClassification = new HierarchicalClassification(customWeakMultiClass, tree);
    final HierarchicalModel model = (HierarchicalModel) hierarchicalClassification.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    printResult(model);
  }

  public void testHierRefinedClass() throws Exception {
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass(iters, step);
    final CustomWeakBinClass customWeakBinClass = new CustomWeakBinClass(iters, step);
    final HierarchicalRefinedClassification hierarchicalRefinedClassification = new HierarchicalRefinedClassification(customWeakBinClass, customWeakMultiClass, tree);
    final HierarchicalModel model = (HierarchicalModel) hierarchicalRefinedClassification.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    printResult(model);
  }
}
