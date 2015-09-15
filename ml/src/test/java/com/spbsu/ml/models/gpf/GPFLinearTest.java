package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.models.gpf.weblogmodel.BlockV1;
import com.spbsu.ml.models.gpf.weblogmodel.WebLogV1GPFSession;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.zip.GZIPInputStream;


import static junit.framework.Assert.assertEquals;

/**
 * User: irlab
 * Date: 22.05.14
 */
@Ignore
public class GPFLinearTest {
  private final int random_seed = 0;

  @Test
  public void testArtificialClicks() throws IOException {
    final List<Session<BlockV1>> dataset_nonfinal;
    try (InputStream is = new GZIPInputStream(WebLogV1GPFSession.class.getResourceAsStream("ses_100k_simple_rand1_h10k.dat.gz"))) {
      dataset_nonfinal = WebLogV1GPFSession.loadDatasetFromJSON(is, new GPFLinearModel(), 100);
    }
    final List<Session<BlockV1>> dataset = dataset_nonfinal;

    System.out.println("dataset size: " + dataset.size());

    FastRandom rand = new FastRandom(random_seed);

    // generate random model
    final GPFLinearModel model_true = new GPFLinearModel();
    model_true.PRUNE_A_THRESHOLD = 1E-5;
    model_true.trainClickProbability(dataset);
    for (int i = 0; i < model_true.NFEATS; i++)
      model_true.theta.set(i, rand.nextGaussian());

    // generate artificial clicks
    int n_sum_clicks = 0;
    for (int nSes = 0; nSes < dataset.size(); nSes++) {
      //System.out.println("session " + nSes);
      final Session<BlockV1> ses = dataset.get(nSes);
      final List<Integer> click_indexes = new ArrayList<Integer>();
      int state = Session.Q_INDEX;
      int click_s = 0;
      while (state != Session.E_INDEX) {
        final double[] probs = new double[ses.getEdgesFrom(state).length];
        for (int j = 0; j < probs.length; j++)
          probs[j] = model_true.eval_f(ses, state, ses.getEdgesFrom(state)[j], click_s);

        double sum = 0;
        for (int j = 0; j < probs.length; j++)
          sum += probs[j];
        for (int j = 0; j < probs.length; j++)
          probs[j] /= sum;

        //        StringBuffer probs_str = new StringBuffer();
        //        for (int j = 0; j < probs.length; j++)
        //          probs_str.append("" + (j == 0 ? "" : ", ") + "(" + state + "->" + ses.getEdgesFrom(state)[j] + ": " + probs[j] + ")");
        //        System.out.println("  probs: " + probs_str);

        state = ses.getEdgesFrom(state)[rand.nextSimple(new ArrayVec(probs))];
        //System.out.println("  state " + state + " " + ses.getBlock(state));
        click_s = ses.getBlock(state).blockType == Session.BlockType.RESULT && rand.nextDouble() <= model_true.getClickGivenViewProbability(ses.getBlock(state)) ? 1 : 0;
        if (click_s == 1)
          click_indexes.add(state);
      }
      ses.setClick_indexes(ArrayTools.convert(click_indexes.toArray(new Integer[click_indexes.size()])));
      //System.out.println("" + nSes + ": " + click_indexes);
      n_sum_clicks += click_indexes.size();
    }
    System.out.println("clicks generated, avg: " + n_sum_clicks/(float)dataset.size() + " clicks/session, " + dataset.size() + " sessions");

    int nObservations = 0;
    for (final Session ses: dataset)
      nObservations += ses.getClick_indexes().length + 1;
    final int fullds_nobservations_all = nObservations;

    for (int random_seed_local = 1; random_seed_local < 2; random_seed_local++) {
      System.out.println("\n############################################################################");
      System.out.println("random_seed_local =\t" + random_seed_local);
      rand = new FastRandom(random_seed_local);

      // generate random model
      final GPFLinearModel model0 = new GPFLinearModel(model_true);
      for (int i = 0; i < model0.NFEATS; i++)
        model0.theta.set(i, rand.nextGaussian());

      final GPFLinearOptimization optimizer = new GPFLinearOptimization();
      final double model_true_expll = Math.exp(-optimizer.evalDatasetGradientValue(model_true, dataset, false).loglikelihood);
      System.out.println("model_true loglikelihood: " + model_true_expll);
      assertEquals(8.4, model_true_expll, 0.1);

      final long t1 = System.currentTimeMillis();
      final double model0_expll = Math.exp(-optimizer.evalDatasetGradientValue(model0, dataset, false).loglikelihood);
      System.out.println("model0 loglikelihood:     " + model0_expll);
      assertEquals(97.9, model0_expll, 0.1);
      final long t2 = System.currentTimeMillis();
      System.out.println("time loglikelihood eval: " + (t2-t1) + " ms");

      final int iteration_dataset_pass_count = 20;

      optimizer.SGD_BLOCK_SIZE = 1;
      final int iteration_count = iteration_dataset_pass_count * dataset.size() / optimizer.SGD_BLOCK_SIZE;
      optimizer.step_eta0 = 0.1; //0.01;
      optimizer.step_gamma = 0.75;
      optimizer.step_a = dataset.size() / optimizer.SGD_BLOCK_SIZE;
      model0.PRUNE_A_THRESHOLD = model_true.PRUNE_A_THRESHOLD;

      System.out.println("optimizer.SGD_BLOCK_SIZE = " + optimizer.SGD_BLOCK_SIZE);
      System.out.println("optimizer.step_eta0      = " + optimizer.step_eta0);
      System.out.println("optimizer.step_a         = " + optimizer.step_a);
      System.out.println("optimizer.step_gamma     = " + optimizer.step_gamma);
      System.out.println("model0.PRUNE_A_THRESHOLD = " + model0.PRUNE_A_THRESHOLD);

      optimizer.listener = new GPFLinearOptimization.IterationEventListener() {
        @Override
        public void iterationPerformed(final GPFLinearOptimization.IterationEvent e) {
          if (optimizer.SGD_BLOCK_SIZE < dataset.size() && e.iter % (dataset.size() / optimizer.SGD_BLOCK_SIZE) != 0) return;
          final double model0_dist = Math.sqrt(model0.theta.l2(e.model.theta));
          final double model_true_dist = Math.sqrt(model_true.theta.l2(e.model.theta));
          double fullds_loglikelihood = e.fullds_loglikelihood;
          int fullds_nobservations_correct = e.fullds_nobservations_correct;
          if (fullds_loglikelihood == 0.) {
            final GPFLinearOptimization.DatasetGradientValue gradV = optimizer.evalDatasetGradientValue(e.model, dataset, false);
            fullds_loglikelihood = gradV.loglikelihood;
            fullds_nobservations_correct = gradV.nObservations;
          }
          System.out.println("" + (new Date()) +
                  "\t" + e.iter + "(" + (e.iter * optimizer.SGD_BLOCK_SIZE / dataset.size()) + "/" + (iteration_count * optimizer.SGD_BLOCK_SIZE / dataset.size()) + ")" +
                  "\tL=" + Math.exp(-fullds_loglikelihood) +
                  "\teta=" + e.step_size +
                  "\tmodel0_dist=" + model0_dist +
                  "\tmodel_true_dist=" + model_true_dist +
                  (optimizer.do_ignore_improbable_sessions ? "" : "\timprobable_obs=" + (fullds_nobservations_all - fullds_nobservations_correct) + "(" + (fullds_nobservations_all - fullds_nobservations_correct)/(float)fullds_nobservations_all + ")") +
                  "\tL_partial=" + Math.exp(-e.loglikelihood) +
                  "\tgrad_norm=" + VecTools.norm(e.gradient) +
                  "\tgrad=[" + e.gradient + "]");
        }

        @Override
        public void backstepPerformed(final GPFLinearOptimization.IterationEvent e) {
          System.out.println("  L > last_L: " + Math.exp(-e.fullds_loglikelihood) + " > " + Math.exp(-e.loglikelihood) + ", go back and set a_m = " + optimizer.step_a_m);
        }
      };

      final GPFLinearModel model_optimized = optimizer.StochasticGradientDescent(model0, dataset, iteration_count);

      final long t3 = System.currentTimeMillis();
      System.out.println("time optimization: " + (t3-t2)/1000 + " sec");
      final double model_final_expll = Math.exp(-optimizer.evalDatasetGradientValue(model_optimized, dataset, false).loglikelihood);
      System.out.println("final loglikelihood:      " + model_final_expll);
      assertEquals(10.3, model_final_expll, 0.1);
    }
  }

  @Test
  public void testOptimizeSGD() throws IOException {
    final List<Session<BlockV1>> dataset_nonfinal;
    try (InputStream is = new GZIPInputStream(WebLogV1GPFSession.class.getResourceAsStream("ses_100k_simple_rand1_h10k.dat.gz"))) {
      dataset_nonfinal = WebLogV1GPFSession.loadDatasetFromJSON(is, new GPFLinearModel(), 100);
    }
    final List<Session<BlockV1>> dataset = dataset_nonfinal;
    final List<Session<BlockV1>> test_dataset_nonfinal;
    try (InputStream is = new GZIPInputStream(WebLogV1GPFSession.class.getResourceAsStream("ses_100k_simple_rand2_h10k.dat.gz"))) {
      test_dataset_nonfinal = WebLogV1GPFSession.loadDatasetFromJSON(is, new GPFLinearModel(), 100);
    }
    final List<Session<BlockV1>> test_dataset = test_dataset_nonfinal;

    final boolean test_sorted_clicks_model = false;
    if (test_sorted_clicks_model) {
      System.out.println("test_sorted_clicks_model");
      for (final Session ses: dataset)
        ses.sortUniqueClicks();
      for (final Session ses: test_dataset)
        ses.sortUniqueClicks();
    }

    int nObservations = 0;
    for (final Session ses: dataset)
      nObservations += ses.getClick_indexes().length + 1;
    final int fullds_nobservations_all = nObservations;

    int n_sum_clicks = 0;
    for (final Session ses: dataset)
      n_sum_clicks += ses.getClick_indexes().length;
    System.out.println("dataset size: " + dataset.size() + " sessions, avg " + (n_sum_clicks / (float)dataset.size()) + " clicks/session");

    final FastRandom rand = new FastRandom(random_seed);

    double best_ll = 1111;
    double test_ll = 1111;
    for (int ntry = 0; ntry < 1; ntry++) {
      System.out.println("########################################################\n");
      System.out.println("" + new Date() + ": ntry: " + ntry + "\n");

      // generate random model
      final GPFLinearModel model0 = new GPFLinearModel();
      model0.trainClickProbability(dataset);
      for (int i = 0; i < model0.NFEATS; i++)
        model0.theta.set(i, rand.nextGaussian());

      final GPFLinearOptimization optimizer = new GPFLinearOptimization();

      final long t1 = System.currentTimeMillis();
      final double model0_expll = Math.exp(-optimizer.evalDatasetGradientValue(model0, dataset, false).loglikelihood);
      System.out.println("model0 loglikelihood:     " + model0_expll);
      assertEquals(13.3, model0_expll, 0.1);
      final long t2 = System.currentTimeMillis();
      System.out.println("time loglikelihood eval: " + (t2-t1) + " ms");

      final int iteration_dataset_pass_count = 10;

      optimizer.SGD_BLOCK_SIZE = 1;
      final int iteration_count = iteration_dataset_pass_count * dataset.size() / optimizer.SGD_BLOCK_SIZE;
      optimizer.step_eta0 = 0.1; //0.01;
      optimizer.step_gamma = 0.75;
      optimizer.step_a = dataset.size() / optimizer.SGD_BLOCK_SIZE;
      model0.PRUNE_A_THRESHOLD = 1E-5;

      System.out.println("optimizer.SGD_BLOCK_SIZE = " + optimizer.SGD_BLOCK_SIZE);
      System.out.println("optimizer.step_eta0      = " + optimizer.step_eta0);
      System.out.println("optimizer.step_a         = " + optimizer.step_a);
      System.out.println("optimizer.step_gamma     = " + optimizer.step_gamma);
      System.out.println("model0.PRUNE_A_THRESHOLD = " + model0.PRUNE_A_THRESHOLD);

      optimizer.listener = new GPFLinearOptimization.IterationEventListener() {
        @Override
        public void iterationPerformed(final GPFLinearOptimization.IterationEvent e) {
          final int iterations_per_dataset = dataset.size() / optimizer.SGD_BLOCK_SIZE;
          if (optimizer.SGD_BLOCK_SIZE < dataset.size()) {
            if (e.iter < iterations_per_dataset) {
              return;
              //if (e.iter % (iterations_per_dataset / 20) != 0)
              //  return;
            } else { // e.iter >= iterations_per_dataset
              if (e.iter % (dataset.size() / optimizer.SGD_BLOCK_SIZE) != 0)
                return;
            }
          }

          final double model0_dist = Math.sqrt(model0.theta.l2(e.model.theta));
          double fullds_loglikelihood = e.fullds_loglikelihood;
          int fullds_nobservations_correct = e.fullds_nobservations_correct;
          if (fullds_loglikelihood == 0.) {
            final GPFLinearOptimization.DatasetGradientValue gradV = optimizer.evalDatasetGradientValue(e.model, dataset, false);
            fullds_loglikelihood = gradV.loglikelihood;
            fullds_nobservations_correct = gradV.nObservations;
          }
          double test_dataset_ll = 0;
          if (test_dataset != null)
            test_dataset_ll = optimizer.evalDatasetGradientValue(e.model, test_dataset, false).loglikelihood;
          System.out.println("" + (new Date()) +
                  "\t" + e.iter + "(" + (e.iter * optimizer.SGD_BLOCK_SIZE / dataset.size()) + "/" + (iteration_count * optimizer.SGD_BLOCK_SIZE / dataset.size()) + ")" +
                  "\tL=" + Math.exp(-fullds_loglikelihood) +
                  "\teta=" + e.step_size +
                  (test_dataset != null ? "\ttest_L=" + Math.exp(-test_dataset_ll) : "") +
                  "\tmodel0_dist=" + model0_dist +
                  (optimizer.do_ignore_improbable_sessions ? "" : "\timprobable_obs=" + (fullds_nobservations_all - fullds_nobservations_correct) + "(" + (fullds_nobservations_all - fullds_nobservations_correct)/(float)fullds_nobservations_all + ")") +
                  "\tL_partial=" + Math.exp(-e.loglikelihood) +
                  "\tgrad_norm=" + VecTools.norm(e.gradient) +
                  ""); //"\tgrad=[" + e.gradient + "]");
          if (e.iter % (iterations_per_dataset * 20) == 0)
            System.out.println(">>current model: " + e.model.explainTheta());
        }

        @Override
        public void backstepPerformed(final GPFLinearOptimization.IterationEvent e) {
          System.out.println("  L > last_L: " + Math.exp(-e.fullds_loglikelihood) + " > " + Math.exp(-e.loglikelihood) + ", go back and set a_m = " + optimizer.step_a_m);
        }
      };

      final GPFLinearModel model_optimized = optimizer.StochasticGradientDescent(model0, dataset, iteration_count);

      final long t3 = System.currentTimeMillis();
      System.out.println("time optimization: " + (t3-t2)/1000 + " sec");
      final double ll = Math.exp(-optimizer.evalDatasetGradientValue(model_optimized, dataset, false).loglikelihood);
      System.out.println("final loglikelihood:      " + ll);
      System.out.println("final theta: " + model_optimized.theta );
      System.out.println("final theta explain: " + model_optimized.explainTheta() );

      if (ll < best_ll) {
        best_ll = ll;
        test_ll = Math.exp(-optimizer.evalDatasetGradientValue(model_optimized, test_dataset, false).loglikelihood);
      }
    }

    System.out.println("" + new Date() + ": best ll: " + best_ll);
    assertEquals(5.2, best_ll, 0.1);
    assertEquals(4.9, test_ll, 0.1);
  }

  @Test
  public void testSERPProbs() throws IOException {
    final GPFLinearModel model = new GPFLinearModel();
    final List<Session<BlockV1>> dataset;
    try (InputStream is = new GZIPInputStream(WebLogV1GPFSession.class.getResourceAsStream("ses_100k_simple_rand1_h10k.dat.gz"))) {
      dataset = WebLogV1GPFSession.loadDatasetFromJSON(is, new GPFLinearModel(), 100);
    }

    // init model
    model.trainClickProbability(dataset);
    // optimized sort_clicks
    //String theta_str = "-1.702713106887966 0.6404247678125509 0.8839508435362965 0.21594343210697917 -0.7358391375584755 -0.8042641035860408 0.3583499027340962 -0.11674658767248532 0.14040130919303337 0.03769440360443547 0.03935260864525687 0.02853823412929953 2.9422529205133463 -2.874955306294313 3.9219531435872557 0.15268044240500608 -1.3913468238331568 0.7810782232327959 0.041232789657154746 -0.45027594953466205 -0.9911457338442456 -1.0436641653093275 -1.283091206075993 -1.2334707757320833 -2.0290071795725835 -1.2284048134884975 -0.8402670201797776 0.09780373481660343 -0.6956589612984125 -0.7010852279098979 0.39816008299399064 -0.03645514963018488 -0.7665757899838521 -0.44931334579482907 1.2404606430397838 2.1373765546696415 -2.3185767965067376 -0.3782845023765775";
    // optimized r602_2.out
    final String theta_str = "-0.9205664691357801 0.9041998193447492 1.0046610326248397 0.29671349018552656 -0.18053090095708907 0.1772697097979266 -0.20372762113889378 -0.7347344786004694 -0.590408428912083 -0.7299015246974587 -0.792449157275554 -0.7089522500922206 2.2627922543859196 -3.100817014916263 4.067342185744358 -2.5792603725334557 1.1895147789581328 1.3296377365812424 0.7446332963557005 0.1835711196264189 0.016013162804432185 -0.18441528045214423 -0.6162991227657141 -0.6939594938332577 1.0894364501659024 0.3579520755136945 0.496534915034393 0.4146067640917571 0.3836857168202354 -0.015704278848354097 0.0913408379926171 -0.04884275707431338 -0.12681530930644924 -0.30889371408471994 0.7818935938652342 2.008411165741512 -4.581704099106069 0.7613830127598948";
    final String[] theta_str_arr = theta_str.split(" ");
    final ArrayVec theta = new ArrayVec(model.NFEATS);
    for (int i = 0; i < theta.dim(); i++)
      theta.set(i, Double.parseDouble(theta_str_arr[i]));
    model.theta.assign(theta);

    // init session
    final Session session = new Session();
    final BlockV1[] blocks = new BlockV1[11];
    for (int i = 0; i < blocks.length; i++) {
      blocks[i] = new BlockV1(
              Session.BlockType.RESULT,
              i == 3 ? BlockV1.ResultType.IMAGES : BlockV1.ResultType.WEB,
              i,
              i <= 3 ? BlockV1.ResultGrade.RELEVANT_PLUS : BlockV1.ResultGrade.NOT_ASED);
    }
    final int[] clicks = new int[] {3, 2, 6, 10};
    WebLogV1GPFSession.setSessionData(session, blocks, clicks);

    System.out.println(model.explainTheta() + "\n");
    System.out.println("selected session");
    System.out.println(model.explainSessionProb(session));

    for (int i = 0; i < 5; i++) {
      System.out.println("\n\nsession #" + (i+1));
      System.out.println(model.explainSessionProb(dataset.get(i)));
    }
  }
}
