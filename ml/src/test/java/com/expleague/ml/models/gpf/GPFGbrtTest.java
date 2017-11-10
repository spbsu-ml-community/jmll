package com.expleague.ml.models.gpf;

import com.expleague.commons.random.FastRandom;
import com.expleague.ml.GridTools;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.models.gpf.weblogmodel.BlockV1;
import com.expleague.ml.models.gpf.weblogmodel.SessionV1AttractivenessModel;
import com.expleague.ml.models.gpf.weblogmodel.WebLogV1ClickProbabilityModel;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import com.expleague.ml.models.gpf.weblogmodel.WebLogV1GPFSession;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.function.Consumer;
import java.util.zip.GZIPInputStream;


import static com.expleague.ml.models.gpf.GPFGbrtOptimization.GPFLoglikelihood;
import static org.junit.Assert.assertEquals;

/**
 * User: irlab
 * Date: 11.07.2014
 */
public class GPFGbrtTest {
  @Test
  public void testGbrtOptimization() throws IOException {
    final FastRandom rng = new FastRandom(0);
    final GPFGbrtModel<BlockV1> model = new GPFGbrtModel<>();
    model.setAttractivenessModel(new SessionV1AttractivenessModel());
    model.setClickProbabilityModel(new WebLogV1ClickProbabilityModel());

    final int rows_limit = 100;
    final double step = 0.2;
    final int iterationsCount = 20;
    final int parallel_processors = Runtime.getRuntime().availableProcessors();

    System.out.println("" + new Date() + "\tload dataset");
    final GPFGbrtOptimization.GPFVectorizedDataset learn;
    final GPFGbrtOptimization.GPFVectorizedDataset validate;
    try (InputStream is = new GZIPInputStream(WebLogV1GPFSession.class.getResourceAsStream("ses_100k_simple_rand1_h10k.dat.gz"))) {
      learn = GPFGbrtOptimization.GPFVectorizedDataset.load(is, model, rows_limit);
    }
    try (InputStream is = new GZIPInputStream(WebLogV1GPFSession.class.getResourceAsStream("ses_100k_simple_rand2_h10k.dat.gz"))) {
      validate = GPFGbrtOptimization.GPFVectorizedDataset.load(is, model, rows_limit);
    }

    System.out.println("" + new Date() + "\ttrainClickProbability");
    model.getClickProbabilityModel().trainClickProbability(learn.sessionList);

    System.out.println("" + new Date() + "\tset up boosting");
    System.out.println("" + new Date() + "\tset up boosting, step=\t" + step);
    final GradientBoosting<GPFLoglikelihood> boosting = new GradientBoosting<GPFGbrtOptimization.GPFLoglikelihood>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn, 32), 6), rng), iterationsCount, step);
    final GPFGbrtOptimization.GPFLoglikelihood learn_loss = new GPFGbrtOptimization.GPFLoglikelihood(model, learn, parallel_processors);
    final GPFGbrtOptimization.GPFLoglikelihood validate_loss = new GPFGbrtOptimization.GPFLoglikelihood(model, validate);

    System.out.println("learn dataset:\t" + learn.sfrList.size() + "\tsessions, " + "feature matrix:\t" + learn.data().rows() + " * " + learn.data().columns());

    final Consumer iterationListener = new GPFGbrtOptimization.PrintProgressIterationListener(learn_loss, validate_loss);
    boosting.addListener(iterationListener);

    System.out.println("" + new Date() + "\tstart learn");
    final Ensemble ans = boosting.fit(learn, learn_loss);

    final double exp_learn_loss = Math.exp(-learn_loss.evalAverageLL(ans));
    System.out.println("" + (new Date()) +
                       "\tfinal" +
                       "\tlearnL=" + exp_learn_loss +
                       "\tvalidL=" + Math.exp(-validate_loss.evalAverageLL(ans)));

    assertEquals(5.7, exp_learn_loss, 0.1);
  }

  public static void main(final String[] args) throws Exception {
    new GPFGbrtTest().testGbrtOptimization();
  }
}
