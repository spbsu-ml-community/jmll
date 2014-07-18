package com.spbsu.ml.models.gpf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.*;


import com.spbsu.commons.func.Action;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
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

    int rows_limit = 1000;
    double step = 0.2;
    int parallel_processors = Runtime.getRuntime().availableProcessors();

    System.out.println("" + new Date() + "\tload dataset");
    GPFVectorizedDataset learn = GPFVectorizedDataset.load("./ml/src/test/data/pgmem/f100/ses_100k_simple_rand1.dat.gz", model, rows_limit);
    GPFVectorizedDataset validate = GPFVectorizedDataset.load("./ml/src/test/data/pgmem/f100/ses_100k_simple_rand2.dat.gz", model, rows_limit);

    System.out.println("" + new Date() + "\ttrainClickProbability");
    model.trainClickProbability(learn.sessionList);

    System.out.println("" + new Date() + "\tset up boosting");
    System.out.println("" + new Date() + "\tset up boosting, step=\t" + step);
    GradientBoosting<GPFLoglikelihood> boosting = new GradientBoosting<GPFLoglikelihood>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn, 32), 6), rng), 20000, step);
//    GradientBoosting<GPFLoglikelihood> boosting = new GradientBoosting<GPFLoglikelihood>(new BootstrapOptimization(new GreedyObliviousTree(GridTools.medianGrid(learn, 32), 6), rng), 20, 0.02);
    GPFLoglikelihood learn_loss = new GPFLoglikelihood(model, learn, parallel_processors);
    GPFLoglikelihood validate_loss = new GPFLoglikelihood(model, validate);

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

  static class GPFVectorizedDataset extends VecDataSetImpl {
    final List<Session> sessionList;
    final List<GPFGbrtModel.SessionFeatureRepresentation> sfrList;
    final int[] sessionPositions; // data[sessionPositions[i]] is the first row for session sfrList[i]

    public GPFVectorizedDataset(List<Session> sessionList, List<GPFGbrtModel.SessionFeatureRepresentation> sfrList, int[] sessionPositions, Mx data) {
      super(data, null);
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
      for (int i = 0; i < sessionList.size(); i++) {
        GPFGbrtModel.SessionFeatureRepresentation sfr = sfrList.get(i);
        System.arraycopy(sfr.features.toArray(), 0, data, sessionPositions[i] * model.NFEATS, sfr.features.dim());
      }

      return new GPFVectorizedDataset(sessionList, sfrList, sessionPositions, new VecBasedMx(model.NFEATS, new ArrayVec(data)));
    }
  }

  static class GPFLoglikelihood extends FuncC1.Stub implements TargetFunc {
    final GPFGbrtModel model;
    final GPFVectorizedDataset dataset;
    final ExecutorService executorPool;

    public GPFLoglikelihood(GPFGbrtModel model, GPFVectorizedDataset dataset) {
      this(model, dataset, 1);
    }

    public GPFLoglikelihood(GPFGbrtModel model, GPFVectorizedDataset dataset, int threadCount) {
      this.executorPool = threadCount == 1 ? Executors.newSingleThreadExecutor() : Executors.newFixedThreadPool(threadCount);
      this.model = model;
      this.dataset = dataset;
    }

    @Override
    public Vec gradient(Vec x) {
      if (x.dim() != dataset.data().rows())
        throw new IllegalArgumentException("x.dim() != dataset.data().rows():" + x.dim() + " != " + dataset.data().rows());

      List<Callable<GPFGbrtModel.SessionGradientValue>> tasks = new ArrayList<>(dataset.sfrList.size());
      List<Vec> sessions_f = new ArrayList<>(dataset.sfrList.size());
      for (int i = 0; i < dataset.sfrList.size(); i++) {
        final GPFGbrtModel.SessionFeatureRepresentation sfr = dataset.sfrList.get(i);
        int start = dataset.sessionPositions[i];

        // old version: Vec f = x.sub(start, sfr.f_count);
        // new non-negative version: f = exp(x)
        final ArrayVec f = new ArrayVec(sfr.f_count);
        for (int j = 0; j < sfr.f_count; j++)
          f.set(j, Math.exp(x.get(start + j)));
        sessions_f.add(f);

        tasks.add(new Callable<GPFGbrtModel.SessionGradientValue>() {
          @Override
          public GPFGbrtModel.SessionGradientValue call() throws Exception {
            return model.eval_L_and_dL_df(sfr, true, f);
          }
        });
      }

      ArrayVec ret = new ArrayVec(dataset.data().rows());
      try {
        List<Future<GPFGbrtModel.SessionGradientValue>> result = executorPool.invokeAll(tasks);

        for (int i = 0; i < dataset.sfrList.size(); i++) {
          int start = dataset.sessionPositions[i];
          Vec gradient = result.get(i).get().gradient;
          Vec f = sessions_f.get(i);
          for (int j = 0; j < gradient.dim(); j++)
            ret.set(start + j, gradient.get(j) * f.get(j));
        }
      } catch (InterruptedException|ExecutionException e) {
        throw new RuntimeException(e);
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
      List<Callable<GPFGbrtModel.SessionGradientValue>> tasks = new ArrayList<>(dataset.sfrList.size());
      for (int i = 0; i < dataset.sfrList.size(); i++) {
        final GPFGbrtModel.SessionFeatureRepresentation sfr = dataset.sfrList.get(i);

        // old version: Vec f = x.sub(start, sfr.f_count);
        // new non-negative version: f = exp(x)
        final Vec f = fmodel.transAll(sfr.features);
        if (f.dim() != sfr.f_count) throw new IllegalArgumentException("wrong fmodel: f.dim() != sfr.f_count, " + f.dim() + " != " + sfr.f_count);
        for (int j = 0; j < f.dim(); j++)
          f.set(j, Math.exp(f.get(j)));

        tasks.add(new Callable<GPFGbrtModel.SessionGradientValue>() {
          @Override
          public GPFGbrtModel.SessionGradientValue call() throws Exception {
            return model.eval_L_and_dL_df(sfr, false, f);
          }
        });
      }

      double loglikelihood = 0.;
      int nObservations = 0;
      try {
        List<Future<GPFGbrtModel.SessionGradientValue>> result = executorPool.invokeAll(tasks);

        for (int i = 0; i < dataset.sfrList.size(); i++) {
          GPFGbrtModel.SessionGradientValue ses_grad = result.get(i).get();
          loglikelihood += ses_grad.loglikelihood;
          nObservations += ses_grad.nObservations;
        }
      } catch (InterruptedException|ExecutionException e) {
        throw new RuntimeException(e);
      }

      return loglikelihood / nObservations;
    }

    @Override
    public int dim() {
      return dataset.data().rows();
    }

    @Override
    public DataSet<?> owner() {
      return dataset;
    }
  }

  private static class IterationListener implements ProgressHandler {
    private int index = 0;
    private double learn_min = Double.POSITIVE_INFINITY;
    private double valid_min = Double.POSITIVE_INFINITY;
    private GPFLoglikelihood learn_loss;
    private GPFLoglikelihood validate_loss;

    private IterationListener(GPFLoglikelihood learn_loss, GPFLoglikelihood validate_loss) {
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
