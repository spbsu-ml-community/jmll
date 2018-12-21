package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.factorization.impl.ALS;
import com.expleague.ml.factorization.impl.ElasticNetFactorization;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multiclass.gradfac.GradFacMulticlass;
import com.expleague.ml.methods.multiclass.gradfac.MultiClassColumnBootstrapOptimization;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.loss.L2;

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
    final Factorization factorization;
    switch (method) {
      case "als":
        factorization = new ALS(iters, lambda);
        break;
      case "elasticnet":
        factorization = new ElasticNetFactorization(iters, 1e-2, alpha, lambda);
        break;
      default:
        throw new UnsupportedOperationException();
    }

    final GradFacMulticlass gradFacMulticlass = new GradFacMulticlass(weak, factorization, (Class<? extends L2>) DataTools.targetByName(localName), printErr);
    return enableBootstrap ? new MultiClassColumnBootstrapOptimization(gradFacMulticlass, random, 1.)
                           : gradFacMulticlass;
  }}
