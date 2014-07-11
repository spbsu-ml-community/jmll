package com.spbsu.ml.models.gpf;

import org.junit.Test;

import java.io.IOException;
import java.util.*;

/**
 * User: irlab
 * Date: 11.07.2014
 */
public class GPFTestGbrt {
  private int random_seed = 0;

  @Test
  public void testEvalGradient() throws IOException {
    final List<Session> dataset = GPFData.loadDatasetFromJSON("./ml/tests/data/pgmem/f100/ses_100k_simple_rand1.dat.gz", new GPFLinearModel(), 1000);
    System.out.println("dataset size: " + dataset.size());

    // generate random model
    final GPFGbrtModel model = new GPFGbrtModel();
    model.PRUNE_A_THRESHOLD = 1E-5;
    model.trainClickProbability(dataset);

    GPFGbrtModel.SessionGradientValue grad = new GPFGbrtModel.SessionGradientValue();
    for (Session ses: dataset) {
      GPFGbrtModel.SessionFeatureRepresentation sesf = new GPFGbrtModel.SessionFeatureRepresentation(ses, model);
      GPFGbrtModel.SessionGradientValue ses_grad = model.eval_L_and_dL_df(sesf, true);
      grad.nObservations += ses_grad.nObservations;
      grad.loglikelihood += ses_grad.loglikelihood;
      if (grad.nObservations < 100) {
//        System.out.println("ses:\t" + ses);
        System.out.println("blocks:\t" + ses.getBlocks().length + "\tses.clicks:\t" + Arrays.toString(ses.getClick_indexes()) + "\teLL:\t" + Math.exp(-ses_grad.loglikelihood/ses_grad.nObservations) + "\tgradient:\t" + ses_grad.gradient);
      }
    }
    System.out.println("nObservations:\t" + grad.nObservations);
    System.out.println("loglikelihood:\t" + grad.loglikelihood / grad.nObservations);
    System.out.println("exp(-loglikelihood):\t" + Math.exp(- grad.loglikelihood / grad.nObservations));
  }
}
