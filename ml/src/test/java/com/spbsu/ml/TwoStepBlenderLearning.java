package com.spbsu.ml;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqReader;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.Pool;
import com.spbsu.ml.data.tools.SubPool;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.loss.*;
import com.spbsu.ml.meta.DSItem;
import com.spbsu.ml.meta.FeatureMeta;
import com.spbsu.ml.meta.PoolFeatureMeta;
import com.spbsu.ml.meta.impl.JsonTargetMeta;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import gnu.trove.list.array.TIntArrayList;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import static java.lang.Math.exp;

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
      final PoolFeatureMeta[] features = pool.features();
      final TIntArrayList relevantForFirstStep = new TIntArrayList();
      for (int i = 0; i < features.length; i++) {
        if (features[i].id().startsWith("YWeb@"))
          relevantForFirstStep.add(i);
      }
      final int[] queryFeatures = relevantForFirstStep.toArray();
      final VecDataSet queryOnlyLearn = learn.joinFeatures(queryFeatures, learn.data());
      final VecDataSet queryOnlyTest = test.joinFeatures(queryFeatures, test.data());
      final Action<Trans> learnTracker = new TransAction("Learn", queryOnlyLearn, learn.target(LLLogit.class));
      final Action<Trans> testTracker = new TransAction("Test", queryOnlyTest, test.target(LLLogit.class));
      final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new GreedyObliviousTree<L2>(GridTools.medianGrid(queryOnlyLearn, 32), 6), LOOL2.class, 2000, 0.01);
      boosting.addListener(learnTracker);
      boosting.addListener(testTracker);
      final Ensemble phaseOneModel = boosting.fit(queryOnlyLearn, learn.target(LLLogit.class));
      DataTools.writeModel(phaseOneModel, new File("phase-one.jmll"));

      {
        residual = phaseOneModel.transAll(pool.joinFeatures(queryFeatures, pool.data()).data());
        for (int i = 0; i < residual.length(); i++) {
          residual.set(i, 1. / (1. + exp(-residual.get(i))));
        }
//        scale(residual, -1);
//        append(residual, phaseOneModel.transAll(pool.joinFeatures(queryFeatures, pool.data()).data()));
//        scale(residual, -1);
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
//    System.out.print(" learn residual: " + Math.sqrt(sum2(phaseTwoLearn.target()) / phaseTwoLearn.dim()));
//    System.out.println(" test residual: " + Math.sqrt(sum2(phaseTwoTest.target()) / phaseTwoTest.dim()));

    {
      final BFGrid grid = GridTools.medianGrid(learn.vecData(), 32);
      StreamTools.transferData(new CharSeqReader(DataTools.SERIALIZATION.write(grid)), new FileWriter("phase-two.grid"));

      final TargetFunc phaseTwoLearn = new ExclusiveComplementLLLogit(0.5, (Vec)learn.target(0), (Vec)learn.target(PHASE_TWO_TARGET), learn.data());
      final TargetFunc phaseTwoTest = new ExclusiveComplementLLLogit(0.5, (Vec)test.target(0), (Vec)test.target(PHASE_TWO_TARGET), test.data());

      final Action<Trans> learnTracker = new TransAction("Learn", learn.vecData(), phaseTwoLearn);
      final Action<Trans> testTracker = new TransAction("Test", test.vecData(), phaseTwoTest);
      final GradientBoosting<TargetFunc> boosting = new GradientBoosting<>(new GreedyObliviousTree<L2>(grid, 6), LOOL2.class, 3000, 0.02);
      boosting.addListener(learnTracker);
      boosting.addListener(testTracker);
      final Ensemble phaseTwoModel = boosting.fit(learn.vecData(), phaseTwoLearn);
      DataTools.writeModel(phaseTwoModel, new File("phase-two.jmll"));
    }
  }

  private static class TransAction implements Action<Trans> {
    private final String message;
    private final Vec cursor;
    private final VecDataSet ds;
    private final Func metric;
    private int index = 0;
    private final int step = 10;

    private TransAction(String message, VecDataSet ds, Func metric) {
      this.message = message;
      this.ds = ds;
      this.metric = metric;
      cursor = new ArrayVec(ds.length());
    }

    @Override
    public void invoke(Trans partial) {
      if (partial instanceof Ensemble) {
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        for (int i = 0; i < ds.length(); i++) {
          if (increment instanceof Ensemble) {
            cursor.adjust(i, linear.wlast() * (increment.trans(ds.data().row(i)).get(0)));
          } else {
            cursor.adjust(i, linear.wlast() * ((Func) increment).value(ds.data().row(i)));
          }
        }
      } else {
        for (int i = 0; i < ds.length(); i++) {
          cursor.set(i, ((Func) partial).value(ds.data().row(i)));
        }
      }

      if (++index % step == 0) {
//        System.out.println(index);
        System.out.println(index + " " + message + " " + metric.getClass().getSimpleName() + ":" + metric.value(cursor));
      }
    }
  }
}
