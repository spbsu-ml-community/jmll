package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.factorization.OuterFactorization;
import com.spbsu.ml.factorization.impl.ALS;
import com.spbsu.ml.factorization.impl.SVDAdapterEjml;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multiclass.GradFacMulticlass;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class MultiClassSplitGradFacBuilder implements Factory<GradFacMulticlass> {
  private final Factory<VecOptimization> defaultWeakBuilder = new BootstrapOptimizationBuilder();

  private VecOptimization weak;
  private String localName = "SatL2";
  private int alsIters = 15;
  private double alsLambda = 0.0;
  private String method = "als";

  public void setWeak(final VecOptimization weak) {
    this.weak = weak;
  }

  public void setIters(final int alsIters) {
    this.alsIters = alsIters;
  }

  public void setLambda(final double alsLambda) {
    this.alsLambda = alsLambda;
  }

  public void setLocal(final String localName) {
    this.localName = localName;
  }

  public void setMethod(final String method) {
    this.method = method;
  }

  @Override
  public GradFacMulticlass create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    final OuterFactorization factorization;
    if (method.equals("svd")) {
      factorization = new SVDAdapterEjml();
    } else {
      factorization = new ALS(alsIters, alsLambda);
    }
    return new GradFacMulticlass(weak, factorization, (Class<? extends L2>) DataTools.targetByName(localName));
  }}
