package com.expleague.ml.cli.builders.methods.impl;

import com.expleague.commons.func.Factory;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.factorization.Factorization;
import com.expleague.ml.factorization.impl.ALS;
import com.expleague.ml.factorization.impl.SVDAdapterEjml;
import com.expleague.ml.loss.L2;
import com.expleague.ml.methods.VecOptimization;
import com.expleague.ml.methods.multiclass.gradfac.GradFacBootstrapMulticlass;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class MultiClassSplitGradFacBootstrapBuilder implements Factory<GradFacBootstrapMulticlass> {
  private final Factory<VecOptimization> defaultWeakBuilder = new BootstrapOptimizationBuilder();

  private VecOptimization weak;
  private String localName = "SatL2";
  private int alsIters = 15;
  private double alsLambda = 0.0;
  private String method = "als";
  private boolean printErr = false;

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

  public void setOut(final boolean printErr) {
    this.printErr = printErr;
  }

  @Override
  public GradFacBootstrapMulticlass create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    final Factorization factorization;
    if (method.equals("als")) {
      factorization = new ALS(alsIters, alsLambda);
    } else {
      factorization = new SVDAdapterEjml();
    }
    return new GradFacBootstrapMulticlass(weak, factorization, (Class<? extends L2>) DataTools.targetByName(localName), printErr);
  }}
