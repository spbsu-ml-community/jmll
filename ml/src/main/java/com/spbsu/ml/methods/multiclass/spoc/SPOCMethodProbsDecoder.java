package com.spbsu.ml.methods.multiclass.spoc;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.Trans;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.loss.LLLogit;
import com.spbsu.ml.loss.blockwise.BlockwiseMLLLogit;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.models.multiclass.MulticlassCodingMatrixModelProbsDecoder;

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
  protected Trans createModel(final Func[] binClass, final VecDataSet learnDS, final BlockwiseMLLLogit llLogit) {
    final CMLMetricOptimization metricOptimization = new CMLMetricOptimization(learnDS, llLogit, S, metricC, metricStep);
    final Mx mu = metricOptimization.trainProbs(codeMatrix, binClass);
    return new MulticlassCodingMatrixModelProbsDecoder(codeMatrix, binClass, MX_IGNORE_THRESHOLD, mu);
  }
}
