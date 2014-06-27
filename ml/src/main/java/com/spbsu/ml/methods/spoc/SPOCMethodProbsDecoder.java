package com.spbsu.ml.methods.spoc;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.ml.Func;
import com.spbsu.ml.Trans;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.models.MulticlassCodingMatrixModelProbsDecoder;

/**
 * User: qdeee
 * Date: 23.05.14
 */
public class SPOCMethodProbsDecoder extends SPOCMethodClassic {
  public static final double METRIC_STEP = 0.05;
  public static final int METRIC_ITERS = 100;
  public static final double METRIC_C = 0.5;
  private final Mx S;

  private final double metricStep;
  private final int metricIters;
  private final double metricC;

  public SPOCMethodProbsDecoder(final Mx codingMatrix, final Mx mxS, final double mcStep, final int mcIters) {
    this(codingMatrix, mxS, mcStep, mcIters, METRIC_STEP, METRIC_ITERS, METRIC_C);
  }

  public SPOCMethodProbsDecoder(final Mx codingMatrix, final Mx mxS, final double mcStep, final int mcIters, final double metricStep, final int metricIters, final double metricC) {
    super(codingMatrix, mcStep, mcIters);
    S = mxS;
    this.metricStep = metricStep;
    this.metricIters = metricIters;
    this.metricC = metricC;
  }

  @Override
  protected Trans createModel(final Func[] binClass, final DataSet learnDS) {
    final CMLMetricOptimization metricOptimization = new CMLMetricOptimization(learnDS, S, metricC, metricIters, metricStep);
    final Mx mu = metricOptimization.trainProbs(codingMatrix, binClass);
    return new MulticlassCodingMatrixModelProbsDecoder(codingMatrix, binClass, MX_IGNORE_THRESHOLD, mu);
  }
}
