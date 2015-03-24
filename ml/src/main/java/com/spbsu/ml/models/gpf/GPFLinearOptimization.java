package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.models.gpf.weblogmodel.BlockV1;


import java.util.*;

/**
 * User: irlab
 * Date: 16.05.14
 */
public class GPFLinearOptimization {
  int SGD_BLOCK_SIZE = 100;
  double step_eta0 = 0.01;
  double step_a = 20;
  double step_gamma = 0.75;
  double step_a_m = 1.;
  double min_eta = 1E-8;
  boolean do_normalize_gradient = false;
  boolean do_random_step = false;

  boolean do_step_decrease_if_worsening = true;
  double step_decrease_if_worsening = 0.25;
  boolean do_ignore_improbable_sessions = false;

//  double GRADIENT_NORM_TOLERANCE = 1E-6;
//  double PERPLEXITY_TOLERANCE = 1E-6;
//  double evalOptimalEta_step = 1.2;

  IterationEventListener listener;

  public GPFLinearModel StochasticGradientDescent(final GPFLinearModel model0, final List<Session<BlockV1>> dataset, final int nIterations) {
    if (dataset.size() < SGD_BLOCK_SIZE)
      throw new IllegalArgumentException("dataset.size() < SGD_BLOCK_SIZE: dataset.size()=" + dataset.size() + ", SGD_BLOCK_SIZE=" + SGD_BLOCK_SIZE);

    final GPFLinearModel model = new GPFLinearModel(model0);
    final Random random = new Random(1);
    final List<Session<BlockV1>> dataset_shuffled = new ArrayList<>(dataset);
    Collections.shuffle(dataset_shuffled, random);
    int dataset_position = 0;
    final ArrayVec last_theta = new ArrayVec(model.theta.toArray());
    double last_ll = Double.NEGATIVE_INFINITY;
    for (int iter = 0; iter < nIterations; iter++) {
      List<Session<BlockV1>> dataset_chunk = null;
      double dataset_ll = 0;
      int fullds_nobservations_correct = 0;
      if (dataset_position + SGD_BLOCK_SIZE <= dataset.size()) {
        dataset_chunk = dataset_shuffled.subList(dataset_position, dataset_position + SGD_BLOCK_SIZE);
        dataset_position += SGD_BLOCK_SIZE;
      } else {
        dataset_chunk = new ArrayList<>(dataset_shuffled.subList(dataset_position, dataset_shuffled.size()));
        Collections.shuffle(dataset_shuffled, random);
        dataset_position = SGD_BLOCK_SIZE - dataset_chunk.size();
        dataset_chunk.addAll(dataset_shuffled.subList(0, dataset_position));

        if (do_step_decrease_if_worsening) {
          final DatasetGradientValue gradV = evalDatasetGradientValue(model, dataset, false);
          dataset_ll = gradV.loglikelihood;
          fullds_nobservations_correct = gradV.nObservations;
          if (dataset_ll > last_ll) {
            // improvement on the whole dataset
            last_ll = dataset_ll;
            last_theta.assign(model.theta);
          } else {
            // go back on the last good theta
            model.theta.assign(last_theta);
            step_a_m *= step_decrease_if_worsening;
            final double eta = step_eta0 * Math.pow(step_a_m * step_a / (step_a/step_a_m + iter), step_gamma);
            if (listener != null)
              listener.backstepPerformed(new IterationEvent(this, iter, model, last_ll, null, 0, eta, dataset_ll, fullds_nobservations_correct));
          }
        }
      }

      final double eta = step_eta0 * Math.pow(step_a_m * step_a / (step_a/step_a_m + iter), step_gamma);
      if (eta < min_eta) {
        return model;
      }

      final DatasetGradientValue gradV = evalDatasetGradientValue(model, dataset_chunk, true);

      // do visualization stuff
      if (listener != null)
        listener.iterationPerformed(new IterationEvent(this, iter, model, gradV.loglikelihood, gradV.gradient, gradV.nObservations, eta, dataset_ll, fullds_nobservations_correct));

      if (gradV.nObservations == 0)
        continue; // improbable session

      // evaluate new_theta
      final ArrayVec new_theta = new ArrayVec(model.NFEATS);
      final double gradient_norm = VecTools.norm(gradV.gradient);
      if (gradient_norm > 0) {
        new_theta.assign(gradV.gradient);

        if (do_normalize_gradient)
          new_theta.scale( 1./gradient_norm );
      } else if (do_random_step) {
        // random direction (distribution is non-uniform, but it is ok)
        for (int i = 0; i < new_theta.dim(); i++)
          new_theta.set(i, random.nextDouble() - 0.5);
        new_theta.scale( 1./VecTools.norm(new_theta) );
      } else {
        // don't do random step
        if (SGD_BLOCK_SIZE >= dataset.size()) {
          return model;
        } else {
          // just try next subset of data
          continue;
        }
      }
      new_theta.scale(eta);
      new_theta.add(model.theta);

      // normalize new_theta
      //new_theta.scale( 1./ new_theta.euclidNorm() );

      model.theta.assign(new_theta);
    }

    return model;
  }

  public void eval_step_params(final double eta_step_0, final double eta_step_T1, final int T1, final double eta_step_T2, final int T2) {
    this.step_eta0 = eta_step_0;
    this.step_gamma = Math.log(eta_step_T1 / eta_step_T2) / Math.log( T2 / T1 );
    this.step_a = Math.exp( Math.log(eta_step_T1 / eta_step_0) / step_gamma + Math.log( T1 ) );
  }

  public void eval_step_params_gamma(final double eta_step_0, final double step_gamma, final double eta_step_T2, final int T2) {
    this.step_eta0 = eta_step_0;
    this.step_gamma = step_gamma;
    this.step_a = Math.pow(eta_step_T2 / eta_step_0, 1/step_gamma) * T2;
  }

//  DatasetGradientValue evalOptimalEta(GPFLinearModel model0, final List<Session> dataset, DatasetGradientValue p0, final double eta_0) {
//    assert eta_0 > 0;
//
//    DatasetGradientValue best_perplexity = new DatasetGradientValue();
//    best_perplexity.perplexity = p0.perplexity;
//    best_perplexity.nObservations = p0.nObservations;
//    best_perplexity.eta = 0.;
//    if (Math.sqrt(p0.gradient.l2(new ArrayVec(model0.NFEATS))) < GRADIENT_NORM_TOLERANCE) {
//      // gradient too small
//      return best_perplexity;
//    }
//
//    int nstep = 0;
//    GPFLinearModel model1 = new GPFLinearModel(model0);
//
//    // first step
//    double eta = eta_0;
//    GPFLinearModel model1 = new GPFLinearModel(model0);
//    ArrayVec step1 = new ArrayVec(p0.gradient.toArray());
//    step1.scale(eta);
//    model1.theta.add(step1);
//    double perplexity = evalDatasetGradientValue(new_model, dataset, false).perplexity;
//    nstep++;
//
//    int direction = 1;
//    if (perplexity > p0.perplexity) {
//      direction = -1;
//    }
//
//    for (; nstep < 100; nstep++) {
//      double new_eta = eta * (direction > 0 ? evalOptimalEta_step : 1. / evalOptimalEta_step);
//      step.assign(p0.gradient);
//      step.scale(new_eta);
//      new_model.theta.assign(model0.theta);
//      new_model.theta.add(step);
//      double new_perplexity = evalDatasetGradientValue(new_model, dataset, false).perplexity;
//      if (new_perplexity > perplexity) {
//        // ухудшение
//        if (nstep == 1 && direction == 1) {
//          // на первом шаге выбрали неправильный direction, разворачиваемся
//        }
//
//      }
//
//
//    }
//
//
//
//    return null;
//  }

  class DatasetGradientValue {
    double loglikelihood = 0.;
    ArrayVec gradient;
    int nObservations = 0;
  }

  DatasetGradientValue evalDatasetGradientValue(final GPFLinearModel model, final List<Session<BlockV1>> dataset, final boolean do_eval_gradient) {
    final DatasetGradientValue ret = new DatasetGradientValue();
    if (do_eval_gradient)
      ret.gradient = new ArrayVec(model.NFEATS);

    for (final Session ses: dataset) {
      final GPFLinearModel.GradientValue gradientValue = model.eval_L_and_Gradient(ses, do_eval_gradient);

      if (do_ignore_improbable_sessions) {
        boolean is_improbable_session = false;
        for (int i = 0; i < gradientValue.observation_probabilities.dim(); i++)
          is_improbable_session = is_improbable_session || gradientValue.observation_probabilities.get(i) < 1E-100;
        if (is_improbable_session) {
          //System.out.println("!!! improbable session: observation_probabilities=" + gradientValue.observation_probabilities + "\n" + ses);
          continue;
        }
      }

      ret.nObservations += gradientValue.observation_probabilities.dim();
      for (int i = 0; i < gradientValue.observation_probabilities.dim(); i++)
        ret.loglikelihood += Math.log(gradientValue.observation_probabilities.get(i));
      if (do_eval_gradient) {
        for (int i = 0; i < gradientValue.observation_probabilities.dim(); i++)
          ret.gradient.add((ArrayVec)gradientValue.gradient.row(i));
      }
    }

    if (ret.nObservations > 0) {
      ret.loglikelihood /= ret.nObservations;
      if (do_eval_gradient)
        ret.gradient.scale(1./ret.nObservations);
    }

    return ret;
  }

  public static class IterationEvent extends EventObject {
    public int iter;
    public GPFLinearModel model;
    public double loglikelihood;
    public ArrayVec gradient;
    public int nObservations = 0;
    public double step_size;
    public double fullds_loglikelihood;
    public int fullds_nobservations_correct;

    public IterationEvent(final Object source) {
      super(source);
    }

    public IterationEvent(final Object source, final int iter, final GPFLinearModel model, final double loglikelihood, final ArrayVec gradient, final int nObservations, final double step_size, final double fullds_loglikelihood, final int fullds_nobservations_correct) {
      super(source);
      this.iter = iter;
      this.model = model;
      this.loglikelihood = loglikelihood;
      this.gradient = gradient;
      this.nObservations = nObservations;
      this.step_size = step_size;
      this.fullds_loglikelihood = fullds_loglikelihood;
      this.fullds_nobservations_correct = fullds_nobservations_correct;
    }
  }

  public interface IterationEventListener {
    public void iterationPerformed(IterationEvent e);
    public void backstepPerformed(IterationEvent e);
  }
}
