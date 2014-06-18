package com.spbsu.ml;

import com.spbsu.commons.filters.Filter;
import com.spbsu.commons.func.Action;
import com.spbsu.commons.func.Computable;
import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecIterator;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.basis.IntBasis;
import com.spbsu.commons.math.vectors.impl.mx.VecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.math.vectors.impl.vectors.SparseVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.logging.Interval;
import com.spbsu.ml.data.impl.DataSetImpl;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.methods.PGMEM;
import com.spbsu.ml.models.pgm.CompositePGM;
import com.spbsu.ml.models.pgm.ProbabilisticGraphicalModel;
import com.spbsu.ml.models.pgm.Route;
import com.spbsu.ml.models.pgm.SimplePGM;
import junit.framework.TestCase;

/**
 * User: solar
 * Date: 26.11.12
 * Time: 15:50
 */
public class PGMEMTest extends TestCase {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public void testPGMFit3x3() {
    SimplePGM original = new SimplePGM(new VecBasedMx(3, new ArrayVec(new double[]{
            0, 0.2, 0.8,
            0, 0, 1.,
            0, 0, 0
    })));
    checkRestoreFixedTopology(original, PGMEM.MOST_PROBABLE_PATH, 0.0, 10, 0.01);
  }

  public void testPGMFit5x5() {
    SimplePGM original = new SimplePGM(new VecBasedMx(5, new ArrayVec(new double[]{
            0, 0.2, 0.3,  0.1,  0.4,
            0, 0,   0.25, 0.25, 0.5,
            0, 0,   0,    0.1,  0.9,
            0, 0,   0.5,  0,    0.5,
            0, 0,   0,    0,    0
    })));


    checkRestoreFixedTopology(original, PGMEM.MOST_PROBABLE_PATH, 0., 100, 0.01);
  }

  public void testPGMFit5x5RandSkip() {
    final SimplePGM original = new SimplePGM(new VecBasedMx(5, new ArrayVec(new double[]{
            0, 0.2, 0.3,  0.1,  0.4,
            0, 0,   0.25, 0.25, 0.5,
            0, 0,   0,    0.1,  0.9,
            0, 0,   0.5,  0,    0.5,
            0, 0,   0,    0,    0
    })));

    checkRestoreFixedTopology(original, PGMEM.LAPLACE_PRIOR_PATH, 0.8, 100, 0.05);
  }
  public void testPGMFit10x10Rand() {
    final VecBasedMx originalMx = new VecBasedMx(10, new ArrayVec(100));
    for (int i = 0; i < originalMx.rows() - 1; i++) {
      for (int j = 0; j < originalMx.columns(); j++)
        originalMx.set(i, j, rng.nextDouble() < 0.5 && j > 0 ? 1 : 0);
      VecTools.normalizeL1(originalMx.row(i));
    }
    VecTools.fill(originalMx.row(originalMx.rows() - 1), 0);
    SimplePGM original = new SimplePGM(originalMx);
    checkRestoreFixedTopology(original, PGMEM.LAPLACE_PRIOR_PATH, 0.5, 100, 0.01);
  }

  public void testPGMFit10x10FreqBasedPriorRand() {
    final VecBasedMx originalMx = new VecBasedMx(10, new ArrayVec(100));
    for (int i = 0; i < originalMx.rows() - 1; i++) {
      for (int j = 0; j < originalMx.columns(); j++)
        originalMx.set(i, j, rng.nextDouble() < 0.5 && j > 0 ? 1 : 0);
      VecTools.normalizeL1(originalMx.row(i));
    }
    VecTools.fill(originalMx.row(originalMx.rows() - 1), 0);
    SimplePGM original = new SimplePGM(originalMx);
    checkRestoreFixedTopology(original, PGMEM.FREQ_DENSITY_PRIOR_PATH, 0.5, 100, 0.01);
  }

  private Vec breakV(Vec next, double lossProbab) {
    Vec result = new SparseVec<IntBasis>(new IntBasis(next.dim()));
    final VecIterator it = next.nonZeroes();
    int resIndex = 0;
    while (it.advance()) {
      if (rng.nextDouble() > lossProbab)
        result.set(resIndex++, it.value());
    }
    return result;
  }

  private void checkRestoreFixedTopology(final SimplePGM original, Computable<ProbabilisticGraphicalModel, PGMEM.Policy> policy, double lossProbab, int iterations, double accuracy) {
    Vec[] ds = new Vec[100000];
    for (int i = 0; i < ds.length; i++) {
      Vec vec;
//      do {
      vec = breakV(original.next(rng), lossProbab);
//      }
//      while (VecTools.norm(vec) < MathTools.EPSILON);
      ds[i] = vec;
    }
    final DataSetImpl dataSet = new DataSetImpl(new VecArrayMx(ds), new ArrayVec(ds.length));
    final VecBasedMx topology = new VecBasedMx(original.topology.columns(), VecTools.fill(new ArrayVec(original.topology.dim()), 1.));
    VecTools.scale(topology.row(topology.rows() - 1), 0.);
    final PGMEM pgmem = new PGMEM(topology, 0.2, iterations, rng, policy);

    final Action<SimplePGM> listener = new Action<SimplePGM>() {
      int iteration = 0;
      @Override
      public void invoke(SimplePGM pgm) {
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
    final SimplePGM fit = pgmem.fit(dataSet, new LLLogit(VecTools.fill(new ArrayVec(dataSet.power()), 1.)));
    VecTools.fill(fit.topology.row(fit.topology.rows() - 1), 0);
    System.out.println(MxTools.prettyPrint(fit.topology));
    System.out.println();
    System.out.println(MxTools.prettyPrint(original.topology));

    assertTrue(VecTools.distance(fit.topology, original.topology) < accuracy * fit.topology.dim());
  }

  public void testCompositeRouteGen() {
    CompositePGM cpgm = new CompositePGM(new VecBasedMx(4, new ArrayVec(
            0., 0.5, 0.5, 0.,
            0, 0, 0.5, 0.3,
            0, 0.5, 0., 0.5,
            0, 0, 0, 0
    )), new CompositePGM.PGMExtensionBinding(1, new SimplePGM(new VecBasedMx(4, new ArrayVec(
            0, 0.3, 0.3, 0.3,
            0, 0, 0.5, 0.5,
            0, 0, 0, 0,
            0, 0, 0, 0
    ))), new int[]{2, 3}));
    final FastRandom rng = new FastRandom(0);
    {
      final Route next = cpgm.next(rng);
      assertEquals("(2,3)->0.25", next.toString());
    }
    {
      final Route next = cpgm.next(rng);
      assertEquals("(1,(3)->0.33333333333333337,3)->0.16666666666666669", next.toString());
    }
    {
      final Route next = cpgm.next(rng);
      assertEquals("(1,(2)->0.33333333333333337,2,1,(2)->0.33333333333333337,2,3)->0.013888888888888892", next.toString());
    }
  }

  public void testCompositeRouteVisit() {
    CompositePGM cpgm = new CompositePGM(new VecBasedMx(4, new ArrayVec(
            0., 0.5, 0.5, 0.,
            0, 0, 0.5, 0.3,
            0, 0.5, 0., 0.5,
            0, 0, 0, 0
    )), new CompositePGM.PGMExtensionBinding(1, new SimplePGM(new VecBasedMx(4, new ArrayVec(
            0, 0.3, 0.3, 0.3,
            0, 0, 0.5, 0.5,
            0, 0, 0, 0,
            0, 0, 0, 0
    ))), new int[]{2, 3}));
    final int[] counter = new int[]{0};
    cpgm.visit(new Filter<Route>() {
      @Override
      public boolean accept(Route route) {
//        System.out.println(route.toString());
        counter[0]++;
        return false;
      }
    });
    assertEquals(197, counter[0]);
  }
}


