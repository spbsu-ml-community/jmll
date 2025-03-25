package com.expleague.ml;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.impl.BFGridImpl;
import com.expleague.ml.impl.BFRowImpl;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.ModelTools;
import com.expleague.ml.models.ObliviousTree;
import com.expleague.ml.testUtils.TestResourceLoader;
import junit.framework.TestCase;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Stream;

/**
 * User: solar
 * Date: 28.04.14
 * Time: 13:18
 */
public class ModelsTest extends GridTest {
  public void testOTlvl2Compile() {
    final BFGrid grid = new BFGridImpl(new BFRowImpl[]{
            new BFRowImpl(0, 0, new double[]{1, 2, 3, 4, 5}),
            new BFRowImpl(5, 1, new double[]{1, 2, 3, 4, 5}),
    });
    final FuncEnsemble<ObliviousTree> sample = new FuncEnsemble<ObliviousTree>(Arrays.asList(new ObliviousTree(
            Arrays.asList(grid.row(0).bf(0), grid.row(1).bf(0)),
            new double[]{1, 2, 3, 4}
    )), 1);

    assertEquals(1., sample.value(new ArrayVec(1, 1)));
    assertEquals(2., sample.value(new ArrayVec(1, 2)));
    assertEquals(3., sample.value(new ArrayVec(2, 1)));
    assertEquals(4., sample.value(new ArrayVec(2, 2)));

    final Func compile = ModelTools.compile(sample);

    assertEquals(1., compile.value(new ArrayVec(1, 1)));
    assertEquals(2., compile.value(new ArrayVec(1, 2)));
    assertEquals(3., compile.value(new ArrayVec(2, 1)));
    assertEquals(4., compile.value(new ArrayVec(2, 2)));
  }

  public void testOTlvl3Compile() {
    final BFGrid grid = new BFGridImpl(new BFRowImpl[]{
            new BFRowImpl(0, 0, new double[]{1}),
            new BFRowImpl(1, 1, new double[]{1}),
            new BFRowImpl(2, 2, new double[]{1}),
    });
    final FuncEnsemble<ObliviousTree> sample = new FuncEnsemble<ObliviousTree>(Arrays.asList(new ObliviousTree(
            Arrays.asList(grid.row(0).bf(0), grid.row(1).bf(0), grid.row(2).bf(0)),
            new double[]{1, 2, 3, 4, 5, 6, 7, 8}
    )), 1);

    assertEquals(1., sample.value(new ArrayVec(1, 1, 1)));
    assertEquals(2., sample.value(new ArrayVec(1, 1, 2)));
    assertEquals(3., sample.value(new ArrayVec(1, 2, 1)));
    assertEquals(4., sample.value(new ArrayVec(1, 2, 2)));
    assertEquals(5., sample.value(new ArrayVec(2, 1, 1)));
    assertEquals(6., sample.value(new ArrayVec(2, 1, 2)));
    assertEquals(7., sample.value(new ArrayVec(2, 2, 1)));
    assertEquals(8., sample.value(new ArrayVec(2, 2, 2)));

    final Func compile = ModelTools.compile(sample);

    assertEquals(1., compile.value(new ArrayVec(1, 1, 1)));
    assertEquals(2., compile.value(new ArrayVec(1, 1, 2)));
    assertEquals(3., compile.value(new ArrayVec(1, 2, 1)));
    assertEquals(4., compile.value(new ArrayVec(1, 2, 2)));
    assertEquals(5., compile.value(new ArrayVec(2, 1, 1)));
    assertEquals(6., compile.value(new ArrayVec(2, 1, 2)));
    assertEquals(7., compile.value(new ArrayVec(2, 2, 1)));
    assertEquals(8., compile.value(new ArrayVec(2, 2, 2)));
  }

  public void testOTlvl4Compile() {
    final BFGrid grid = new BFGridImpl(new BFRowImpl[]{
            new BFRowImpl(0, 0, new double[]{1}),
            new BFRowImpl(1, 1, new double[]{1}),
            new BFRowImpl(2, 2, new double[]{1}),
            new BFRowImpl(3, 3, new double[]{1}),
    });
    final FuncEnsemble<ObliviousTree> sample = new FuncEnsemble<ObliviousTree>(Arrays.asList(new ObliviousTree(
            Arrays.asList(
                    grid.row(0).bf(0),
                    grid.row(1).bf(0),
                    grid.row(2).bf(0),
                    grid.row(3).bf(0)),
            new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    )), 1);

    assertEquals(1., sample.value(new ArrayVec(1, 1, 1, 1)));
    assertEquals(2., sample.value(new ArrayVec(1, 1, 1, 2)));
    assertEquals(3., sample.value(new ArrayVec(1, 1, 2, 1)));
    assertEquals(4., sample.value(new ArrayVec(1, 1, 2, 2)));
    assertEquals(5., sample.value(new ArrayVec(1, 2, 1, 1)));
    assertEquals(6., sample.value(new ArrayVec(1, 2, 1, 2)));
    assertEquals(7., sample.value(new ArrayVec(1, 2, 2, 1)));
    assertEquals(8., sample.value(new ArrayVec(1, 2, 2, 2)));
    assertEquals(9., sample.value(new ArrayVec(2, 1, 1, 1)));
    assertEquals(10., sample.value(new ArrayVec(2, 1, 1, 2)));
    assertEquals(11., sample.value(new ArrayVec(2, 1, 2, 1)));
    assertEquals(12., sample.value(new ArrayVec(2, 1, 2, 2)));
    assertEquals(13., sample.value(new ArrayVec(2, 2, 1, 1)));
    assertEquals(14., sample.value(new ArrayVec(2, 2, 1, 2)));
    assertEquals(15., sample.value(new ArrayVec(2, 2, 2, 1)));
    assertEquals(16., sample.value(new ArrayVec(2, 2, 2, 2)));

    final Func compile = ModelTools.compile(sample);

    assertEquals(1., compile.value(new ArrayVec(1, 1, 1, 1)));
    assertEquals(2., compile.value(new ArrayVec(1, 1, 1, 2)));
    assertEquals(3., compile.value(new ArrayVec(1, 1, 2, 1)));
    assertEquals(4., compile.value(new ArrayVec(1, 1, 2, 2)));
    assertEquals(5., compile.value(new ArrayVec(1, 2, 1, 1)));
    assertEquals(6., compile.value(new ArrayVec(1, 2, 1, 2)));
    assertEquals(7., compile.value(new ArrayVec(1, 2, 2, 1)));
    assertEquals(8., compile.value(new ArrayVec(1, 2, 2, 2)));
    assertEquals(9., compile.value(new ArrayVec(2, 1, 1, 1)));
    assertEquals(10., compile.value(new ArrayVec(2, 1, 1, 2)));
    assertEquals(11., compile.value(new ArrayVec(2, 1, 2, 1)));
    assertEquals(12., compile.value(new ArrayVec(2, 1, 2, 2)));
    assertEquals(13., compile.value(new ArrayVec(2, 2, 1, 1)));
    assertEquals(14., compile.value(new ArrayVec(2, 2, 1, 2)));
    assertEquals(15., compile.value(new ArrayVec(2, 2, 2, 1)));
    assertEquals(16., compile.value(new ArrayVec(2, 2, 2, 2)));
  }

  public void testOTCompleteEnsemble() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
            new BootstrapOptimization<>(
                    new GreedyObliviousTree<>(GridTools.medianGrid(learn.vecData(), 32), 6),
                    new FastRandom(100500)),
            L2Reg.class, 200, 0.005
    );
    //noinspection unchecked
    final Ensemble<ObliviousTree> fit = boosting.fit(learn.vecData(), learn.target(SatL2.class));
    final ModelTools.CompiledOTEnsemble compiled = ModelTools.compile(fit);
    VecDataSet validateVecs = validate.vecData();
    validateVecs.<Stream<Vec>>stream().forEach(vec -> {
      assertEquals(compiled.value(vec), fit.trans(vec).get(0), 1e-5);
//      for (int i = 0; i < fit.size(); i++) {
//        System.out.print(i + " ");
//
//        final ObliviousTree[] trees = new ObliviousTree[i + 1];
//        for (int j = 0; j <= i; j++) {
//          trees[j] = fit.model(j);
//        }
//        Ensemble<ObliviousTree> subEnsamble = new Ensemble<>(trees, fit.weights().sub(0, trees.length));
//        final ModelTools.CompiledOTEnsemble compiled = ModelTools.compile(subEnsamble);
//        if (Math.abs(compiled.value(vec) - subEnsamble.trans(vec).get(0)) > 1e-5) {
//          ModelTools.compile(new Ensemble<>(new ObliviousTree[]{fit.model(i)}, fit.weights().sub(0, 1)));
//          assertTrue(false);
//        }
//        assertEquals(compiled.value(vec), subEnsamble.trans(vec).get(0), 1e-5);
//      }
//      System.out.println();
    });
  }
}