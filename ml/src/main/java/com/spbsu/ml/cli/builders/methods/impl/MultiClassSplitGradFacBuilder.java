package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.factorization.OuterFactorization;
import com.spbsu.ml.factorization.impl.ALS;
import com.spbsu.ml.factorization.impl.ElasticNetFactorization;
import com.spbsu.ml.factorization.impl.SVDAdapterEjml;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multiclass.gradfac.GradFacMulticlass;
import com.spbsu.ml.methods.multiclass.gradfac.MultiClassColumnBootstrapOptimization;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class MultiClassSplitGradFacBuilder implements Factory<VecOptimization> {
  public static FastRandom defaultRandom;

  private final Factory<VecOptimization> defaultWeakBuilder = new BootstrapOptimizationBuilder();

  private VecOptimization weak;
  private String localName = "SatL2";

  private String method = "als";
  private int iters = 20;
  private double lambda = 0.0;
  private double alpha = 0.95;

  private boolean enableBootstrap = false;
  private FastRandom random = defaultRandom;

  private boolean printErr = false;

  public void setWeak(final VecOptimization weak) {
    this.weak = weak;
  }

  public void setIters(final int alsIters) {
    this.iters = alsIters;
  }

  public void setLambda(final double alsLambda) {
    this.lambda = alsLambda;
  }

  public void setLocal(final String localName) {
    this.localName = localName;
  }

  public void setMethod(final String method) {
    this.method = method;
  }

  public void setOut(final boolean printErr) {
    this.printErr = printErr;
  }

  public void setBootstrap(final boolean enable) {
    this.enableBootstrap = enable;
  }

  @Override
  public VecOptimization create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    final OuterFactorization factorization;
    switch (method) {
      case "als":
        factorization = new ALS(iters, lambda);
        break;
      case "elasticnet":
        factorization = new ElasticNetFactorization(iters, 1e-2, alpha, lambda);
        break;
      default:
        factorization = new SVDAdapterEjml();
        break;
    }

    final GradFacMulticlass gradFacMulticlass = new GradFacMulticlass(weak, factorization, (Class<? extends L2>) DataTools.targetByName(localName), printErr);
    return enableBootstrap ? new MultiClassColumnBootstrapOptimization(gradFacMulticlass, random, 1.)
                           : gradFacMulticlass;
  }}
