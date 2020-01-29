package com.expleague.gregion;

import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.MxTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.*;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.L2GreedyTDRegion;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.SatL2;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.LassoGradientBoosting;
import com.expleague.ml.methods.LassoRegionsForest;
import com.expleague.ml.methods.greedyRegion.*;
import com.expleague.otboost.ObliviousTreeBoostingTest;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class GreedyRegionTest extends GridTest {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public void testGreedyTDRegionLassoBoost() {
    final int N = 1000;
    final LassoGradientBoosting<L2> boosting = new LassoGradientBoosting<>(
        new BootstrapOptimization<>(new GreedyTDRegion<>(GridTools.medianGrid(learn.vecData(), 32)), rng),
        L2GreedyTDRegion.class, N
    );
//                    new GreedyTDIterativeRegion(GridTools.medianGrid(learn.vecData(), 32)), rng), L2GreedyTDRegion.class, N);
    boosting.setLambda(1e-4);
    final L2 target = learn.target(L2.class);
    final LassoProgressPrinter progressPrinter = new LassoProgressPrinter(target, validate.vecData(), validate.target(L2.class),N);
    boosting.addListener(progressPrinter);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  public void testGRBoost() {
    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(new GreedyRegion(new FastRandom(), GridTools.medianGrid(learn.vecData(), 32)), rng),
        L2.class, 10000, 0.02
    );
    final Consumer counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void accept(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final L2 target = learn.target(L2.class);
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), target);
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    final Consumer<Trans> modelPrinter = new ModelPrinter();
    final Consumer<Trans> qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  public void testGTDRForestBoost() {
    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new RegionForest<>(GridTools.medianGrid(learn.vecData(), 32), rng, 5),
        L2GreedyTDRegion.class, 12000, 0.004
    );
    final Consumer<Trans> counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void accept(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), learn.target(L2.class));
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    final Consumer<Trans> modelPrinter = new ModelPrinter();
    final Consumer<Trans> qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }


  public void testGTDRFLassoBoost() {
    double val = Double.valueOf("1.1754944e-38");

    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new LassoRegionsForest<>(new GreedyTDRegion<>(GridTools.medianGrid(learn.vecData(), 32)), rng, 5),
        L2GreedyTDRegion.class, 12000,0.01
    );
//    (new LassoRegionsForest<L2>(new GreedyTDIterativeRegion<WeightedLoss<? extends L2>>(GridTools.medianGrid(learn.vecData(), 32)), rng,1), L2GreedyTDRegion.class, 12000,1);
    final Consumer counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void accept(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), learn.target(L2.class));
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    final Consumer<Trans> modelPrinter = new ModelPrinter();
    final Consumer<Trans> qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  public void testGTDRBoost() {
//    final GradientBoosting<L2> boosting = new GradientBoosting
//            (new BootstrapOptimization<>(
//                         new GreedyMergedRegion(GridTools.medianGrid(learn.vecData(), 32)), rng), L2GreedyTDRegion.class, 12000, 0.07);
    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new GreedyTDRegion<>(GridTools.medianGrid(learn.vecData(), 32), 6, 1e-4),
        L2GreedyTDRegion.class, 3000, 0.015
    );
    final Consumer<Trans> counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void accept(Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), learn.target(L2.class));
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    final Consumer<Trans> modelPrinter = new ModelPrinter();
    final Consumer<Trans> qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  public void testGreedyTDLinearRegionBoost() {
    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(new GreedyTDLinearRegion<>(GridTools.medianGrid(learn.vecData(), 32), 6, 1e-3), rng),
        L2GreedyTDRegion.class, 1000, 0.015
    );
    final Consumer<Trans> counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void accept(Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), learn.target(L2.class));
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    final Consumer<Trans> modelPrinter = new ModelPrinter();
    final Consumer<Trans> qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  public void testProbRegionBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<>(new BootstrapOptimization<>(new GreedyProbLinearRegion<>(GridTools.medianGrid(learn.vecData(), 32), 6), rng), L2Reg.class, 5000, 0.1);
    new ObliviousTreeBoostingTest.addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
  }

  public void testGTDRIBoost() {
    final GradientBoosting<L2> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(new GreedyTDIterativeRegion<>(GridTools.medianGrid(learn.vecData(), 32)), rng),
        L2GreedyTDRegion.class, 12000, 0.002
    );
    final Consumer counter = new ProgressHandler() {
      int index = 0;

      @Override
      public void accept(Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer("\tlearn:\t", learn.vecData(), learn.target(L2.class));
    final ScoreCalcer validateListener = new ScoreCalcer("\ttest:\t", validate.vecData(), validate.target(L2.class));
    final Consumer<Trans> modelPrinter = new ModelPrinter();
    final Consumer<Trans> qualityCalcer = new QualityCalcer();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
    boosting.addListener(qualityCalcer);
//    boosting.addListener(modelPrinter);
    boosting.fit(learn.vecData(), learn.target(L2.class));
  }

  class LassoProgressPrinter implements Consumer<LassoGradientBoosting.LassoGBIterationResult> {
    private int iterations = 0;
    private L2 learnTarget;
    private L2 testTarget;
    private VecDataSet test;
    private List<Vec> transformedTest;

    public LassoProgressPrinter(L2 loss, VecDataSet validate, L2 validateTarget, int iterations) {
      this.learnTarget = loss;
      this.test = validate;
      this.testTarget = validateTarget;
      this.transformedTest = new ArrayList<>(iterations);
    }
    double min = Double.POSITIVE_INFINITY;
    double testMin = Double.POSITIVE_INFINITY;
    @Override
    public void accept(LassoGradientBoosting.LassoGBIterationResult iterationResult) {
      double curLoss = VecTools.distance(iterationResult.cursor, learnTarget.target) / Math.sqrt(learnTarget.target.dim());
      System.out.print(iterations + " learn score: " + curLoss);
      min = Math.min(curLoss, min);
      System.out.print(" minimum = " + min);
      ++iterations;
      Vec applied = iterationResult.addedModel.transAll(test.data()).col(0);
      VecTools.scale(applied, -1.0);
      transformedTest.add(applied);
      if (iterations % 10 == 0) {
        Mx currentTest = new ColsVecArrayMx(transformedTest.toArray(new Vec[transformedTest.size()]));
        Vec testCursor = MxTools.multiply(currentTest,iterationResult.newWeights.sub(0,iterations));
        double testLoss = VecTools.distance(testCursor, testTarget.target) / Math.sqrt(testTarget.target.dim());
        System.out.print(iterations + " test score: " + testLoss);
        testMin = Math.min(testLoss, testMin);
        System.out.print(" minimum = " + testMin);
      }
      System.out.println();
    }
  }
}
