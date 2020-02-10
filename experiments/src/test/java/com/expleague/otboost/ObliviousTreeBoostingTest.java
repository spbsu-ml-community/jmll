package com.expleague.otboost;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.*;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.*;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.*;
import com.expleague.ml.methods.*;
import com.expleague.ml.methods.greedyRegion.*;
import com.expleague.ml.methods.trees.GreedyObliviousLinearTree;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.ObliviousTree;

import java.util.Random;
import java.util.function.Consumer;

import static com.expleague.commons.math.MathTools.sqr;
import static com.expleague.commons.math.vectors.VecTools.copy;
import static java.lang.Math.exp;
import static java.lang.Math.log;

/**
 * User: solar
 * Date: 26.11.12
 *
 * Time: 15:50
 */
@SuppressWarnings("RedundantArrayCreation")
public class ObliviousTreeBoostingTest extends GridTest {
  private FastRandom rng;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    rng = new FastRandom(0);
  }

  public static class addBoostingListeners<GlobalLoss extends TargetFunc> {
    public addBoostingListeners(final GradientBoosting<GlobalLoss> boosting, final GlobalLoss loss, final Pool<?> _learn, final Pool<?> _validate) {
      final Consumer counter = new ProgressHandler() {
        int index = 0;
        @Override
        public void accept(final Trans partial) {
          System.out.print("\n" + index++);
        }
      };
      final ScoreCalcer learnListener = new ScoreCalcer(/*"\tlearn:\t"*/"\t", _learn.vecData(), _learn.target(L2.class));
      final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", _validate.vecData(), _validate.target(L2.class));
      final Consumer<Trans> modelPrinter = new ModelPrinter();
      boosting.addListener(counter);
      boosting.addListener(learnListener);
      boosting.addListener(validateListener);
      boosting.addListener(modelPrinter);
      final Ensemble ans = boosting.fit(_learn.vecData(), loss);
      Vec current = new ArrayVec(_validate.size());
      for (int i = 0; i < _validate.size(); i++) {
        double f = 0;
        for (int j = 0; j < ans.size(); j++)
          f += ans.weight(j) * ((Func) ans.model(j)).value(_validate.vecData().data().row(i));
        current.set(i, f);
      }
      System.out.println("\n + Final loss = " + VecTools.distance(current, _validate.target(L2.class).target) / Math.sqrt(_validate.size()));
    }
  }

  public void testOTBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(
        new BootstrapOptimization<>(
            new GreedyObliviousTree<>(GridTools.medianGrid(learn.vecData(), 32), 6),
        rng),
        L2Reg.class, 2000, 0.005
    );
    new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
  }

  public void testOTLinearBoost() {
    final GradientBoosting<SatL2> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(
            new GreedyObliviousLinearTree<>(GridTools.medianGrid(learn.vecData(), 32), 6),
          rng),
        L2.class, 2000, 0.002
    );
    new addBoostingListeners<>(boosting, learn.target(SatL2.class), learn, validate);
  }

  public void testClassifyBoost() {
    final ProgressHandler pl = new ProgressHandler() {
      public static final int WSIZE = 10;
      final Vec cursor = new ArrayVec(learn.size());
      final double[] window = new double[WSIZE];
      int index = 0;
      double h_t = entropy(cursor);

      private double entropy(Vec cursor) {
        double result = 0;
        for (int i = 0; i < learn.target(0).length(); i++) {
          final double pX;
          if ((Double)learn.target(0).at(i) > 0)
            pX = 1./(1. + exp(-cursor.get(i)));
          else
            pX = 1./(1. + exp(cursor.get(i)));
          result += - pX * log(pX) / log(2);
        }
        return result;
      }

      @Override
      public void accept(Trans partial) {
        if (!(partial instanceof Ensemble))
          throw new RuntimeException("Can not work with other than ensembles");
        final Ensemble linear = (Ensemble) partial;
        final Trans increment = linear.last();
        Vec inc = copy(cursor);
        for (int i = 0; i < learn.size(); i++) {
          if (increment instanceof Ensemble) {
            cursor.adjust(i, linear.wlast() * (increment.trans(learn.vecData().data().row(i)).get(0)));
            inc.adjust(i, increment.trans(learn.vecData().data().row(i)).get(0));
          } else {
            cursor.adjust(i, linear.wlast() * ((Func) increment).value(learn.vecData().data().row(i)));
            inc.adjust(i, ((Func) increment).value(learn.vecData().data().row(i)));
          }
        }
        double h_t1 = entropy(inc);
        final double score = (h_t - h_t1) / info(increment);
        window[index % window.length] = score;
        index++;
        double meanScore = 0;
        for (int i = 0; i < window.length; i++)
          meanScore += Math.max(0, window[i]/ window.length);
        System.out.println("Info score: " + score + " window score: " + meanScore);
        h_t = entropy(cursor);
      }

      private double info(Trans increment) {
        if (increment instanceof ObliviousTree) {
          double info = 0;
          double total = 0;
          final double[] based = ((ObliviousTree) increment).based();
          for(int i = 0; i < based.length; i++) {
            final double fold = based[i];
            info += (fold + 1) * log(fold + 1);
            total += fold;
          }
          info -= log(total + based.length);
          info /= -(total + based.length);
          return info;
        }
        return Double.POSITIVE_INFINITY;
      }
    };

//    Action<Trans> learnScore = new ScorePrinter("Learn", learn.vecData(), learn.target(LLLogit.class));
//    Action<Trans> testScore = new ScorePrinter("Test", validate.vecData(), validate.target(LLLogit.class));
    Consumer<Trans> iteration = new Consumer<Trans>() {
      int index = 0;
      @Override
      public void accept(Trans trans) {
        System.out.println(index++);
      }
    };
    final GradientBoosting<LLLogit> boosting = new GradientBoosting<>(
        new BootstrapOptimization<>(new GreedyObliviousTree<>(GridTools.medianGrid(learn.vecData(), 32), 6), rng),
        LOOL2.class, 2000, 0.05
    );
    boosting.addListener(iteration);
//    boosting.addListener(pl);
//    boosting.addListener(learnScore);
//    boosting.addListener(testScore);
    boosting.fit(learn.vecData(), learn.target(LLLogit.class));
  }
}


