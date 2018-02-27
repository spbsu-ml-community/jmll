package com.expleague.ml;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.SubPool;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.LOOL2;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.commons.io.StreamTools;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.CharSeqReader;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.meta.DSItem;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.PoolFeatureMeta;
import com.expleague.ml.meta.impl.JsonTargetMeta;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import gnu.trove.list.array.TIntArrayList;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.function.Consumer;

import static com.expleague.commons.math.vectors.VecTools.scale;

/**
 * User: solar
 * Date: 18.03.15
 * Time: 17:23
 */
public class TwoStepBlenderLearning {
  public static final String PHASE_TWO_TARGET = "PhaseTwoTarget";

  public static void main(String[] args) throws IOException {
    final FastRandom rnd = new FastRandom(0);
//    final Pool<? extends DSItem> pool = DataTools.loadFromFile(args[0]);
    final Pool<? extends DSItem> pool = DataTools.loadFromFile("/Users/solar/data/pools/blender/nov/pool-long-click-probability");
    final int[][] cvSplit = DataTools.splitAtRandom(pool.size(), rnd, 0.2, 0.8);
    Pool<? extends DSItem> learn = new SubPool<>(pool, cvSplit[0]);
    Pool<? extends DSItem> test = new SubPool<>(pool, cvSplit[1]);

    System.out.println("Phase 1");
    final File resFile = new File("phase-one.residual");
    final Vec residual;
    if (!resFile.exists()) {
//    {
      final PoolFeatureMeta[] features = pool.features();
      final TIntArrayList relevantForFirstStep = new TIntArrayList();
      for (int i = 0; i < features.length; i++) {
        if (features[i].id().startsWith("YWeb@"))
          relevantForFirstStep.add(i);
      }
      final int[] queryFeatures = relevantForFirstStep.toArray();
      final L2 phaseOneTarget = learn.target(L2.class);
      final VecDataSet queryOnlyLearn = learn.joinFeatures(queryFeatures, learn.data());
      final VecDataSet queryOnlyTest = test.joinFeatures(queryFeatures, test.data());
      final Consumer<Trans> learnTracker = new ScorePrinter("Learn", queryOnlyLearn, phaseOneTarget);
      final Consumer<Trans> testTracker = new ScorePrinter("Test", queryOnlyTest, test.target(L2.class));
      final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new GreedyObliviousTree<L2>(GridTools.medianGrid(queryOnlyLearn, 32), 6), LOOL2.class, 2000, 0.005);
      boosting.addListener(learnTracker);
      boosting.addListener(testTracker);
      final Ensemble phaseOneModel = boosting.fit(queryOnlyLearn, phaseOneTarget);
//      final Ensemble phaseOneModel = boosting.fit(queryOnlyLearn, learn.target(LLLogit.class));
      DataTools.writeModel(phaseOneModel, new File("phase-one.jmll"));

      {
        residual = phaseOneModel.transAll(pool.joinFeatures(queryFeatures, pool.data()).data());
//        for (int i = 0; i < residual.length(); i++) {
//          residual.set(i, 1. / (1. + exp(-residual.get(i))));
//        }
        VecTools.scale(residual, -1);
        VecTools.append(residual, (Vec)pool.target(0));
      }
      StreamTools.transferData(new CharSeqReader(DataTools.SERIALIZATION.write(residual)), new FileWriter("phase-one.residual"));
    }
    else {
      residual = DataTools.SERIALIZATION.read(StreamTools.readFile(resFile), Mx.class);
    }
    {
      JsonTargetMeta meta = new JsonTargetMeta();
      meta.id = PHASE_TWO_TARGET;
      meta.type = FeatureMeta.ValueType.VEC;
      meta.owner = pool;
      meta.associated = pool.data().meta().id();
      pool.addTarget(meta, residual);
      learn = new SubPool<>(pool, cvSplit[0]);
      test = new SubPool<>(pool, cvSplit[1]);
    }

    System.out.println("Phase 2");
//    System.out.print(" learn residual: " + Math.sqrt(sum2(phaseTwoLearn.target()) / phaseTwoLearn.xdim()));
//    System.out.println(" test residual: " + Math.sqrt(sum2(phaseTwoTest.target()) / phaseTwoTest.xdim()));

    {
      final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
      StreamTools.transferData(new CharSeqReader(DataTools.SERIALIZATION.write(grid)), new FileWriter("phase-two.grid"));

//      final TargetFunc phaseTwoLearn = new ExclusiveComplementLLLogit(0.5, (Vec)learn.target(0), (Vec)learn.target(PHASE_TWO_TARGET), learn.data());
//      final TargetFunc phaseTwoTest = new ExclusiveComplementLLLogit(0.5, (Vec)test.target(0), (Vec)test.target(PHASE_TWO_TARGET), test.data());
//      final TargetFunc phaseTwoLearn = new ExclusiveComplementLLLogit(0.5, (Vec)learn.target(0), (Vec)learn.target(PHASE_TWO_TARGET), learn.data());
//      final TargetFunc phaseTwoTest = new ExclusiveComplementLLLogit(0.5, (Vec)test.target(0), (Vec)test.target(PHASE_TWO_TARGET), test.data());
      final TargetFunc phaseTwoLearn = new L2((Vec)learn.target(PHASE_TWO_TARGET), learn.data());
      final TargetFunc phaseTwoTest = new L2((Vec)test.target(PHASE_TWO_TARGET), test.data());

      final Consumer<Trans> learnTracker = new ScorePrinter("Learn", learn.vecData(), phaseTwoLearn);
      final Consumer<Trans> testTracker = new ScorePrinter("Test", test.vecData(), phaseTwoTest);
      final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new GreedyObliviousTree<L2>(grid, 6), LOOL2.class, 3000, 0.005);
      boosting.addListener(learnTracker);
      boosting.addListener(testTracker);
      final Ensemble phaseTwoModel = boosting.fit(learn.vecData(), phaseTwoLearn);
      DataTools.writeModel(phaseTwoModel, new File("phase-two.jmll"));
    }
  }

}
