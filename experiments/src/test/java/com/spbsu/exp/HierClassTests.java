package com.spbsu.exp;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.tree.IntTree;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.HierTools;
import com.spbsu.ml.data.tools.MCTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.L2;
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

      learn.addTarget(new FakeTargetMeta(learn.vecData(), FeatureMeta.ValueType.INTS), mapTarget(learnTarget, map));
      test.addTarget(new FakeTargetMeta(test.vecData(), FeatureMeta.ValueType.INTS), mapTarget(testTarget, map));
    }
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    init();
  }

  public void testHierClass() throws Exception {
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass();
    final HierarchicalClassification hierarchicalClassification = new HierarchicalClassification(customWeakMultiClass, tree);
    final HierarchicalModel model = (HierarchicalModel) hierarchicalClassification.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    MCTools.evalModel(model, learn, "[LEARN]");
    MCTools.evalModel(model, test, "[TEST]");
  }

  public void testHierRefinedClass() throws Exception {
    final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass();
    final CustomWeakBinClass customWeakBinClass = new CustomWeakBinClass();
    final HierarchicalRefinedClassification hierarchicalRefinedClassification = new HierarchicalRefinedClassification(customWeakBinClass, customWeakMultiClass, tree);
    final HierarchicalModel model = (HierarchicalModel) hierarchicalRefinedClassification.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    MCTools.evalModel(model, learn, "[LEARN]");
    MCTools.evalModel(model, test, "[TEST]");
  }

  private static class CustomWeakMultiClass extends VecOptimization.Stub<TargetFunc> {

    private final int iters;
    private final double step;

    public CustomWeakMultiClass() {
      iters = 100;
      step = 1.5;
    }

    @Override
    public Trans fit(final VecDataSet learn, final TargetFunc loss) {
      final BFGrid grid = GridTools.medianGrid(learn, 32);
      final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), SatL2.class), iters, step);

      final String comment = prepareComment(((BlockwiseMLLLogit) loss).labels());
      final ProgressHandler calcer = new ProgressHandler() {
        int iter = 0;

        @Override
        public void invoke(Trans partial) {
          if ((iter + 1) % 20 == 0) {
            if (((Ensemble) partial).last() instanceof MultiClassModel) {
              final MultiClassModel model = MCTools.joinBoostingResults((Ensemble) partial);
              final Mx x = model.transAll(learn.data());
              double value = loss.value(x);
              System.out.print(comment + ", iter=" + iter + ", [learn]MLLLogitValue=" + value + "\r");
            }
          }
          iter++;
        }
      };
      boosting.addListener(calcer);
      final Ensemble ensemble = boosting.fit(learn, loss);
      final MCModel model = MCTools.joinBoostingResults(ensemble);
      System.out.println();
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

  private static class CustomWeakBinClass extends VecOptimization.Stub<TargetFunc> {
    @Override
    public Trans fit(final VecDataSet learn, final TargetFunc targetFunc) {
      final CustomWeakMultiClass customWeakMultiClass = new CustomWeakMultiClass();
      final MultiClassModel mcm = (MultiClassModel) customWeakMultiClass.fit(learn, targetFunc);
      return mcm.getInternModel().dirs()[0];
    }
  }

  private static IntSeq mapTarget(final IntSeq intTarget, final TIntIntMap mapping) {
    final int[] mapped = new int[intTarget.length()];
    for (int i = 0; i < intTarget.length(); i++) {
      mapped[i] = mapping.get(intTarget.at(i));
    }
    return new IntSeq(mapped);
  }

}
