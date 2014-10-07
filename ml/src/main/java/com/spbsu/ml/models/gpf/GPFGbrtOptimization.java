package com.spbsu.ml.models.gpf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.*;


import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.*;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.func.Ensemble;
import com.spbsu.ml.models.gpf.weblogmodel.BlockV1;
import com.spbsu.ml.models.gpf.weblogmodel.WebLogV1GPFSession;

/**
 * Created with IntelliJ IDEA.
 * User: irlab
 * Date: 16.07.14
 * Time: 11:27
 * To change this template use File | Settings | File Templates.
 */
public class GPFGbrtOptimization {
  public static class GPFVectorizedDataset<Blk extends Session.Block> extends VecDataSetImpl {
    final List<Session<Blk>> sessionList;
    public final List<GPFGbrtModel.SessionFeatureRepresentation<Blk>> sfrList;
    final int[] sessionPositions; // data[sessionPositions[i]] is the first row for session sfrList[i]

    public GPFVectorizedDataset(List<Session<Blk>> sessionList, List<GPFGbrtModel.SessionFeatureRepresentation<Blk>> sfrList, int[] sessionPositions, Mx data) {
      super(data, null);
      this.sessionList = sessionList;
      this.sfrList = sfrList;
      this.sessionPositions = sessionPositions;
    }

    public static GPFVectorizedDataset<BlockV1> load(String filename, GPFGbrtModel<BlockV1> model, int rows_limit) throws IOException {
      List<Session<BlockV1>> sessionList = WebLogV1GPFSession.loadDatasetFromJSON(filename, model, rows_limit);

      List<GPFGbrtModel.SessionFeatureRepresentation<BlockV1>> sfrList = new ArrayList<>(sessionList.size());
      int[] sessionPositions = new int[sessionList.size()];
      int datasetSize = 0;
      for (int i = 0; i < sessionList.size(); i++) {
        GPFGbrtModel.SessionFeatureRepresentation<BlockV1> sfr = new GPFGbrtModel.SessionFeatureRepresentation<>(sessionList.get(i), model);
        sfrList.add(sfr);
        sessionPositions[i] = datasetSize;
        datasetSize += sfr.f_count;
      }
      double[] data = new double[datasetSize * model.getEdgeFeatCount()];
      for (int i = 0; i < sessionList.size(); i++) {
        GPFGbrtModel.SessionFeatureRepresentation sfr = sfrList.get(i);
        System.arraycopy(sfr.features.toArray(), 0, data, sessionPositions[i] * model.getEdgeFeatCount(), sfr.features.dim());
      }

      return new GPFVectorizedDataset<>(sessionList, sfrList, sessionPositions, new VecBasedMx(model.getEdgeFeatCount(), new ArrayVec(data)));
    }
  }

  public static class GPFLoglikelihood<Blk extends Session.Block> extends FuncC1.Stub implements TargetFunc {
    final GPFGbrtModel<Blk> model;
    final GPFVectorizedDataset<Blk> dataset;
    final ExecutorService executorPool;

    private Vec[] fvalue_partial;
    private int[] fvalue_partial_size;

    public GPFLoglikelihood(GPFGbrtModel<Blk> model, GPFVectorizedDataset<Blk> dataset) {
      this(model, dataset, 1);
    }

    public GPFLoglikelihood(GPFGbrtModel<Blk> model, GPFVectorizedDataset<Blk> dataset, int threadCount) {
      this.executorPool = threadCount == 1 ? Executors.newSingleThreadExecutor() : Executors.newFixedThreadPool(threadCount);
      this.model = model;
      this.dataset = dataset;

      this.fvalue_partial = new Vec[dataset.sfrList.size()];
      for (int i = 0; i < dataset.sfrList.size(); i++) {
        fvalue_partial[i] = new ArrayVec(dataset.sfrList.get(i).f_count);
      }
      this.fvalue_partial_size = new int[dataset.sfrList.size()];
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
        Vec f = null;
        if (fmodel instanceof Ensemble) {
          final Ensemble linear = (Ensemble) fmodel;
          if (linear.size() == fvalue_partial_size[i] + 1) {
            final Trans increment = linear.last();
            final double weight_last = linear.wlast();
            VecTools.incscale(fvalue_partial[i], increment.transAll(sfr.features), weight_last);
            fvalue_partial_size[i]++;
            f = fvalue_partial[i];
          } else if (linear.size() == fvalue_partial_size[i]) {
            f = fvalue_partial[i];
          } else {
            f = fmodel.transAll(sfr.features);
            throw new IllegalStateException("unexpected state: linear.size() == " + linear.size() + ", fvalue_partial_size[i] == " + fvalue_partial_size[i] + ", you can safely remove this exception call");
          }
        } else {
          f = fmodel.transAll(sfr.features);
          throw new IllegalStateException("unexpected state: !(fmodel instanceof Ensemble), you can safely remove this exception call");
        }

        if (f.dim() != sfr.f_count) throw new IllegalArgumentException("wrong fmodel: f.dim() != sfr.f_count, " + f.dim() + " != " + sfr.f_count);

        final Vec f_exp = new ArrayVec(f.dim());
        for (int j = 0; j < f.dim(); j++)
          f_exp.set(j, Math.exp(f.get(j)));

        tasks.add(new Callable<GPFGbrtModel.SessionGradientValue>() {
          @Override
          public GPFGbrtModel.SessionGradientValue call() throws Exception {
            return model.eval_L_and_dL_df(sfr, false, f_exp);
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

  public static class PrintProgressIterationListener implements ProgressHandler {
    private int index = 0;
    private double learn_min = Double.POSITIVE_INFINITY;
    private double valid_min = Double.POSITIVE_INFINITY;
    private GPFLoglikelihood learn_loss;
    private GPFLoglikelihood validate_loss;

    public PrintProgressIterationListener(GPFLoglikelihood learn_loss, GPFLoglikelihood validate_loss) {
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
}
