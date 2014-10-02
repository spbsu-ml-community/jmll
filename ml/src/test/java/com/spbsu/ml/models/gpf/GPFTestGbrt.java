package com.spbsu.ml.models.gpf;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import org.junit.Test;

import java.io.IOException;
import java.util.*;

import static com.spbsu.ml.models.gpf.GPFGbrtOptimization.GPFVectorizedDataset;
import static com.spbsu.ml.models.gpf.GPFGbrtOptimization.GPFLoglikelihood;
import static com.spbsu.ml.models.gpf.GPFGbrtOptimization.PrintProgressIterationListener;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * User: irlab
 * Date: 11.07.2014
 */
public class GPFTestGbrt {
  @Test
  public void testGbrtOptimization() throws IOException {
    FastRandom rng = new FastRandom(0);
    GPFGbrtModel model = new GPFGbrtModel();

    int rows_limit = 100;
    double step = 0.2;
    int iterationsCount = 20;
    int parallel_processors = Runtime.getRuntime().availableProcessors();

    System.out.println("" + new Date() + "\tload dataset");
    GPFVectorizedDataset learn = GPFVectorizedDataset.load("./jmll/ml/src/test/data/pgmem/f100/ses_100k_simple_rand1_h10k.dat.gz", model, rows_limit);
    GPFVectorizedDataset validate = GPFVectorizedDataset.load("./jmll/ml/src/test/data/pgmem/f100/ses_100k_simple_rand2_h10k.dat.gz", model, rows_limit);

    System.out.println("" + new Date() + "\ttrainClickProbability");
    model.trainClickProbability(learn.sessionList);

    System.out.println("" + new Date() + "\tset up boosting");
    System.out.println("" + new Date() + "\tset up boosting, step=\t" + step);
    GradientBoosting<GPFLoglikelihood> boosting = new GradientBoosting<GPFGbrtOptimization.GPFLoglikelihood>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn, 32), 6), rng), iterationsCount, step);
    GPFLoglikelihood learn_loss = new GPFLoglikelihood(model, learn, parallel_processors);
    GPFLoglikelihood validate_loss = new GPFLoglikelihood(model, validate);

    System.out.println("learn dataset:\t" + learn.sfrList.size() + "\tsessions, " + "feature matrix:\t" + learn.data().rows() + " * " + learn.data().columns());

    Action iterationListener = new PrintProgressIterationListener(learn_loss, validate_loss);
    boosting.addListener(iterationListener);

    System.out.println("" + new Date() + "\tstart learn");
    Ensemble ans = boosting.fit(learn, learn_loss);

    double exp_learn_loss = Math.exp(-learn_loss.evalAverageLL(ans));
    System.out.println("" + (new Date()) +
                       "\tfinal" +
                       "\tlearnL=" + exp_learn_loss +
                       "\tvalidL=" + Math.exp(-validate_loss.evalAverageLL(ans)));

    assertEquals(exp_learn_loss, 5.7, 0.1);
  }

  public static void main(String[] args) throws Exception {
    new GPFTestGbrt().testGbrtOptimization();
  }
}
