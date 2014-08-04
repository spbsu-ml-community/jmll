package com.spbsu.ml;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.util.tree.FastTree;
import com.spbsu.commons.util.tree.Tree;
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
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.models.HierarchicalModel;
import com.spbsu.ml.models.MCModel;
import com.spbsu.ml.test_utils.TestResourceLoader;
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
  private static Tree tree;

  private synchronized void init() throws Exception {
    if (learn == null || test == null || tree == null) {
      learn = TestResourceLoader.loadPool("features.txt.gz");
      test = TestResourceLoader.loadPool("featuresTest.txt.gz");

      final TDoubleList borders = new TDoubleArrayList();
      final IntSeq learnTarget = MCTools.transformRegressionToMC(learn.target(L2.class).target, 16, borders);
      final IntSeq testTarget = MCTools.transformRegressionToMC(test.target(L2.class).target, borders.size(), borders);

      final Tree sourceTree = HierTools._loadOrderedMulticlassAsHierarchicalMedian(learnTarget);
      tree = HierTools.pruneTree(sourceTree, learnTarget, 450);

      final TIntIntMap map = new TIntIntHashMap();
      HierTools.createTreesMapping(sourceTree.getRoot(), tree.getRoot(), map);

      final IntSeq prunedLearnTarget = mapTarget(learnTarget, map);
      final IntSeq prunedTestTarget = mapTarget(testTarget, map);

      learn.addTarget(new FakeTargetMeta(learn.vecData(), FeatureMeta.ValueType.INTS), prunedLearnTarget);
      test.addTarget(new FakeTargetMeta(test.vecData(), FeatureMeta.ValueType.INTS), prunedTestTarget);
    }
  }


  @Override
  protected void setUp() throws Exception {
    super.setUp();
    init();
  }

  public void testHierClass() throws Exception {
    final SpecialWeakModel specialWeakModel = new SpecialWeakModel();
    final HierarchicalClassification hierarchicalClassification = new HierarchicalClassification(specialWeakModel, (FastTree) tree);
    final HierarchicalModel model = (HierarchicalModel) hierarchicalClassification.fit(learn.vecData(), learn.target(BlockwiseMLLLogit.class));
    MCTools.evalModel(model, learn, "[LEARN]");
    MCTools.evalModel(model, test, "[TEST]");
  }

  private static class SpecialWeakModel extends VecOptimization.Stub<TargetFunc> {
    @Override
    public Trans fit(final VecDataSet learn, final TargetFunc loss) {
      final BFGrid grid = GridTools.medianGrid(learn, 32);
      final GradientBoosting<TargetFunc> weak = new GradientBoosting<>(new MultiClass(new GreedyObliviousTree<L2>(grid, 5), SatL2.class), 100, 0.4);
      final Ensemble ensemble = weak.fit(learn, loss);
      final MCModel model = MCTools.joinBoostingResults(ensemble);
      return model;
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
