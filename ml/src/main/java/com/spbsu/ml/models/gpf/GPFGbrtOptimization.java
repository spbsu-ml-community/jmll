package com.spbsu.ml.models.gpf;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.*;
import com.spbsu.ml.data.impl.LightDataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;

import java.io.IOException;
import java.util.*;

import org.junit.Test;

/**
 * Created with IntelliJ IDEA.
 * User: irlab
 * Date: 16.07.14
 * Time: 11:27
 * To change this template use File | Settings | File Templates.
 */
public class GPFGbrtOptimization {
  @Test
  public void testGbrtOptimization() throws IOException {
    FastRandom rng = new FastRandom(0);
    GPFGbrtModel model = new GPFGbrtModel();

    System.out.println("" + new Date() + "\tload dataset");
    GPFVectorizedDataset learn = GPFVectorizedDataset.load("./ml/src/test/data/pgmem/f100/ses_100k_simple_rand1.dat.gz", model, 1000);
    GPFVectorizedDataset validate = GPFVectorizedDataset.load("./ml/src/test/data/pgmem/f100/ses_100k_simple_rand2.dat.gz", model, 1000);

    System.out.println("" + new Date() + "\ttrainClickProbability");
    model.trainClickProbability(learn.sessionList);

    System.out.println("" + new Date() + "\tset up boosting");
    GradientBoosting<GPFLoss> boosting = new GradientBoosting<GPFLoss>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn, 32), 6), rng), 2000, 0.02);
//    GradientBoosting<GPFLoss> boosting = new GradientBoosting<GPFLoss>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn, 32), 6), rng), 20, 0.02);
    GPFLoss learn_loss = new GPFLoss(model, learn);
    GPFLoss validate_loss = new GPFLoss(model, validate);

    System.out.println("learn dataset:\t" + learn.sfrList.size() + "\tsessions, " + "feature matrix:\t" + learn.data().rows() + " * " + learn.data().columns());

    Action iterationListener = new IterationListener(learn_loss, validate_loss);
    boosting.addListener(iterationListener);

    System.out.println("" + new Date() + "\tstart learn");
    Ensemble ans = boosting.fit(learn, learn_loss);

    System.out.println("" + (new Date()) +
            "\tfinal" +
            "\tlearnL=" + Math.exp( -learn_loss.evalAverageLL(ans) ) +
            "\tvalidL=" + Math.exp( -validate_loss.evalAverageLL(ans)) );
  }

  static class GPFVectorizedDataset extends LightDataSetImpl {
    final List<Session> sessionList;
    final List<GPFGbrtModel.SessionFeatureRepresentation> sfrList;
    final int[] sessionPositions; // data[sessionPositions[i]] is the first row for session sfrList[i]

    public GPFVectorizedDataset(List<Session> sessionList, List<GPFGbrtModel.SessionFeatureRepresentation> sfrList, int[] sessionPositions, double[] data, double[] target) {
      super(data, target);
      this.sessionList = sessionList;
      this.sfrList = sfrList;
      this.sessionPositions = sessionPositions;
    }

    public static GPFVectorizedDataset load(String filename, GPFGbrtModel model, int rows_limit) throws IOException {
      List<Session> sessionList = GPFData.loadDatasetFromJSON(filename, model, rows_limit);

      List<GPFGbrtModel.SessionFeatureRepresentation> sfrList = new ArrayList<GPFGbrtModel.SessionFeatureRepresentation>(sessionList.size());
      int[] sessionPositions = new int[sessionList.size()];
      int datasetSize = 0;
      for (int i = 0; i < sessionList.size(); i++) {
        GPFGbrtModel.SessionFeatureRepresentation sfr = new GPFGbrtModel.SessionFeatureRepresentation(sessionList.get(i), model);
        sfrList.add(sfr);
        sessionPositions[i] = datasetSize;
        datasetSize += sfr.f_count;
      }
      double[] data = new double[datasetSize * model.NFEATS];
      double[] target = new double[datasetSize]; // empty content, zeroes, not used
      for (int i = 0; i < sessionList.size(); i++) {
        GPFGbrtModel.SessionFeatureRepresentation sfr = sfrList.get(i);
        System.arraycopy(sfr.features.toArray(), 0, data, sessionPositions[i] * model.NFEATS, sfr.features.dim());
      }

      return new GPFVectorizedDataset(sessionList, sfrList, sessionPositions, data, target);
    }
  }

  static class GPFLoss extends FuncC1.Stub {
    final GPFGbrtModel model;
    final GPFVectorizedDataset dataset;

    public GPFLoss(GPFGbrtModel model, GPFVectorizedDataset dataset) {
      this.model = model;
      this.dataset = dataset;
    }

    @Override
    public Vec gradient(Vec x) {
      if (x.dim() != dataset.data().rows())
        throw new IllegalArgumentException("x.dim() != dataset.data().rows():" + x.dim() + " != " + dataset.data().rows());
      ArrayVec ret = new ArrayVec(dataset.data().rows());
      for (int i = 0; i < dataset.sfrList.size(); i++) {
        GPFGbrtModel.SessionFeatureRepresentation sfr = dataset.sfrList.get(i);
        int start = dataset.sessionPositions[i];

        // old version: Vec f = x.sub(start, sfr.f_count);
        // new non-negative version: f = exp(x)
        ArrayVec f = new ArrayVec(sfr.f_count);
        for (int j = 0; j < sfr.f_count; j++)
          f.set(j, Math.exp(x.get(start + j)));

        GPFGbrtModel.SessionGradientValue ses_grad = model.eval_L_and_dL_df(sfr, true, f);

        for (int j = 0; j < ses_grad.gradient.dim(); j++)
          ret.set(start + j, ses_grad.gradient.get(j) * f.get(j));
      }

      ret.scale(-1); // optimize for maximization
      return ret;
    }

    @Override
    public double value(Vec x) {
      double loglikelihood = 0.;
      int nObservations = 0;
      for (int i = 0; i < dataset.sfrList.size(); i++) {
        GPFGbrtModel.SessionFeatureRepresentation sfr = dataset.sfrList.get(i);
        int start = dataset.sessionPositions[i];

        // old version: Vec f = x.sub(start, sfr.f_count);
        // new non-negative version: f = exp(x)
        ArrayVec f = new ArrayVec(sfr.f_count);
        for (int j = 0; j < sfr.f_count; j++)
          f.set(j, Math.exp(x.get(start + j)));

        GPFGbrtModel.SessionGradientValue ses_grad = model.eval_L_and_dL_df(sfr, false, f);
        loglikelihood += ses_grad.loglikelihood;
        nObservations += ses_grad.nObservations;
      }
      return -loglikelihood;
    }

    public double evalAverageLL(Trans fmodel) {
      double loglikelihood = 0.;
      int nObservations = 0;
      for (int i = 0; i < dataset.sfrList.size(); i++) {
        GPFGbrtModel.SessionFeatureRepresentation sfr = dataset.sfrList.get(i);

        // old version: Vec f = x.sub(start, sfr.f_count);
        // new non-negative version: f = exp(x)
        Vec f = fmodel.transAll(sfr.features);
        if (f.dim() != sfr.f_count) throw new IllegalArgumentException("wrong fmodel: f.dim() != sfr.f_count, " + f.dim() + " != " + sfr.f_count);
        for (int j = 0; j < f.dim(); j++)
          f.set(j, Math.exp(f.get(j)));

        GPFGbrtModel.SessionGradientValue ses_grad = model.eval_L_and_dL_df(sfr, false, f);
        loglikelihood += ses_grad.loglikelihood;
        nObservations += ses_grad.nObservations;
      }
      return loglikelihood / nObservations;
    }

    @Override
    public int dim() {
      return dataset.data().rows();
    }
  }

  private static class IterationListener implements ProgressHandler {
    private int index = 0;
    private double learn_min = Double.POSITIVE_INFINITY;
    private double valid_min = Double.POSITIVE_INFINITY;
    private GPFLoss learn_loss;
    private GPFLoss validate_loss;

    private IterationListener(GPFLoss learn_loss, GPFLoss validate_loss) {
      this.learn_loss = learn_loss;
      this.validate_loss = validate_loss;
    }

    @Override
    public void invoke(Trans partial) {
      double learn_eL = Math.exp(-learn_loss.evalAverageLL(partial));
      double valid_eL = Math.exp(-validate_loss.evalAverageLL(partial));
      learn_min = Math.min(learn_min, learn_eL);
      valid_min = Math.min(valid_min, valid_eL);
      System.out.println("" + (new Date()) +
              "\t" + (++index) +
              "\tlearnL=" + learn_eL +
              "\tmin_learnL=" + learn_min +
              "\tvalidL=" + valid_eL +
              "\tmin_validL=" + valid_min);
    }
  }

  public static void main(String[] args) throws Exception {
    new GPFGbrtOptimization().testGbrtOptimization();
  }
}
