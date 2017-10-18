package com.expleague.ml.methods.multiclass.spoc;

import com.expleague.commons.math.Func;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.models.multiclass.MulticlassCodingMatrixModelProbsDecoder;

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
  private final double metricC;

  public SPOCMethodProbsDecoder(final Mx codingMatrix, final Mx mxS, final VecOptimization<LLLogit> weak) {
    this(codingMatrix, mxS, weak, METRIC_STEP, METRIC_C);
  }

  public SPOCMethodProbsDecoder(final Mx codingMatrix, final Mx mxS, final VecOptimization<LLLogit> weak, final double metricStep,
                                final double metricC) {
    super(codingMatrix, weak);
    this.S = mxS;
    this.metricStep = metricStep;
    this.metricC = metricC;
  }

  @Override
  protected MulticlassCodingMatrixModelProbsDecoder createModel(final Func[] binClass, final VecDataSet learnDS, final BlockwiseMLLLogit llLogit) {
    final CMLMetricOptimization metricOptimization = new CMLMetricOptimization(learnDS, llLogit, S, metricC, metricStep);
    final Mx mu = metricOptimization.trainProbs(codeMatrix, binClass);
    return new MulticlassCodingMatrixModelProbsDecoder(codeMatrix, binClass, MX_IGNORE_THRESHOLD, mu);
  }
}
