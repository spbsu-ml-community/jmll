package com.expleague.pgm;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecIterator;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.math.vectors.impl.vectors.SparseVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.util.logging.Interval;
import com.expleague.ml.GridTest;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.methods.PGMEM;
import com.expleague.ml.models.pgm.ProbabilisticGraphicalModel;
import com.expleague.ml.models.pgm.Route;
import com.expleague.ml.models.pgm.SimplePGM;

import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertTrue;

@SuppressWarnings("RedundantArrayCreation")
public class PGMTest extends GridTest {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public void testPGMFit5x5RandSkip() {
    final SimplePGM original = new SimplePGM(new VecBasedMx(5, new ArrayVec(new double[]{
        0, 0.2, 0.3, 0.1, 0.4,
        0, 0, 0.25, 0.25, 0.5,
        0, 0, 0, 0.1, 0.9,
        0, 0, 0.5, 0, 0.5,
        0, 0, 0, 0, 0
    })));

    checkRestoreFixedTopology(original, PGMEM.GAMMA_PRIOR_PATH, 0.8, 100, 0.01);
  }

  public void testPGMFit10x10Rand() {
    final VecBasedMx originalMx = new VecBasedMx(10, new ArrayVec(100));
    for (int i = 0; i < originalMx.rows() - 1; i++) {
      for (int j = 0; j < originalMx.columns(); j++)
        originalMx.set(i, j, rng.nextDouble() < 0.5 && j > 0 ? 1 : 0);
      VecTools.normalizeL1(originalMx.row(i));
    }
    VecTools.fill(originalMx.row(originalMx.rows() - 1), 0);
    final SimplePGM original = new SimplePGM(originalMx);
    checkRestoreFixedTopology(original, PGMEM.POISSON_PRIOR_PATH, 0.5, 100, 0.01);
  }

  public void testPGMFit10x10FreqBasedPriorRand() {
    final VecBasedMx originalMx = new VecBasedMx(10, new ArrayVec(100));
    for (int i = 0; i < originalMx.rows() - 1; i++) {
      for (int j = 0; j < originalMx.columns(); j++)
        originalMx.set(i, j, rng.nextDouble() < 0.5 && j > 0 ? 1 : 0);
      VecTools.normalizeL1(originalMx.row(i));
    }
    VecTools.fill(originalMx.row(originalMx.rows() - 1), 0);
    final SimplePGM original = new SimplePGM(originalMx);
    checkRestoreFixedTopology(original, PGMEM.POISSON_PRIOR_PATH, 0.5, 100, 0.01);
  }

  public void testPGMFit3x3() {
    final SimplePGM original = new SimplePGM(new VecBasedMx(3, new ArrayVec(new double[]{
        0, 0.2, 0.8,
        0, 0, 1.,
        0, 0, 0
    })));
    checkRestoreFixedTopology(original, PGMEM.GAMMA_PRIOR_PATH, 0.0, 100, 0.01);
  }

  public void testPGMFit5x5() {
    final SimplePGM original = new SimplePGM(new VecBasedMx(5, new ArrayVec(new double[]{
        0, 0.2, 0.3, 0.1, 0.4,
        0, 0, 0.25, 0.25, 0.5,
        0, 0, 0, 0.1, 0.9,
        0, 0, 0.5, 0, 0.5,
        0, 0, 0, 0, 0
    })));


    checkRestoreFixedTopology(original, PGMEM.GAMMA_PRIOR_PATH, 0., 100, 0.01);
  }

  private Vec breakV(final Vec next, final double lossProbab) {
    final Vec result = new SparseVec(next.dim());
    final VecIterator it = next.nonZeroes();
    int resIndex = 0;
    while (it.advance()) {
      if (rng.nextDouble() > lossProbab)
        result.set(resIndex++, it.value());
    }
    return result;
  }

  private void checkRestoreFixedTopology(final SimplePGM original, final Function<ProbabilisticGraphicalModel, PGMEM.Policy> policy, final double lossProbab, final int iterations, final double accuracy) {
    final Vec[] ds = new Vec[100000];
    for (int i = 0; i < ds.length; i++) {
      //TODO: @solar, please, fix it
      Vec vec;
      do {
        Route route = original.next(rng);
        Vec next = new ArrayVec(route.length());
        IntStream.range(0, route.length()).forEach(idx -> next.set(idx, route.dst(idx)));
        vec = breakV(next, lossProbab);
      }
      while (VecTools.norm(vec) < MathTools.EPSILON);
      ds[i] = vec;
    }
    final VecDataSet dataSet = new VecDataSetImpl(new RowsVecArrayMx(ds), null);
    final VecBasedMx topology = new VecBasedMx(original.topology.columns(), VecTools.fill(new ArrayVec(original.topology.dim()), 1.));
    VecTools.scale(topology.row(topology.rows() - 1), 0.);
    final PGMEM pgmem = new PGMEM(topology, 0.5, iterations, rng, policy);

    final Consumer<SimplePGM> listener = new Consumer<SimplePGM>() {
      int iteration = 0;

      @Override
      public void accept(final SimplePGM pgm) {
        Interval.stopAndPrint("Iteration " + ++iteration);
        System.out.println();
        System.out.print(VecTools.distance(pgm.topology, original.topology));
        for (int i = 0; i < pgm.topology.columns(); i++) {
          System.out.print(" " + VecTools.distance(pgm.topology.row(i), original.topology.row(i)));
        }
        System.out.println();
        Interval.start();
      }
    };
    pgmem.addListener(listener);

    Interval.start();
    final SimplePGM fit = pgmem.fit(dataSet, new LLLogit(VecTools.fill(new ArrayVec(dataSet.length()), 1.), dataSet));
    VecTools.fill(fit.topology.row(fit.topology.rows() - 1), 0);
    System.out.println(MxTools.prettyPrint(fit.topology));
    System.out.println();
    System.out.println(MxTools.prettyPrint(original.topology));

    assertTrue(VecTools.distance(fit.topology, original.topology) < accuracy * fit.topology.dim());
  }
}
