package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.VecOptimization;
import com.spbsu.ml.methods.multiclass.gradfac.GradFacSvdNMulticlass;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class MultiClassSplitGradFacSvdNBuilder implements Factory<GradFacSvdNMulticlass> {
  private final Factory<VecOptimization> defaultWeakBuilder = new BootstrapOptimizationBuilder();

  private VecOptimization weak;
  private String localName = "SatL2";
  private int factorDim = 1;
  private boolean needCompact = true;
  private boolean printErrors = false;
  private boolean bootstrap = false;

  public void setWeak(final VecOptimization weak) {
    this.weak = weak;
  }

  public void setLocal(final String localName) {
    this.localName = localName;
  }

  public void setDim(final int factorDim) {
    this.factorDim = factorDim;
  }

  public void setCompact(final boolean needCompact) {
    this.needCompact = needCompact;
  }

  public void setOut(final boolean printErrors) {
    this.printErrors = printErrors;
  }

  public void setBootstrap(final boolean enable) {
    this.bootstrap = enable;
  }

  @Override
  public GradFacSvdNMulticlass create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    return new GradFacSvdNMulticlass(weak, (Class<? extends L2>) DataTools.targetByName(localName), factorDim, needCompact, printErrors);
  }}
