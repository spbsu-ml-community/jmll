package com.spbsu.ml.cli.builders.methods.impl;

import com.spbsu.commons.func.Factory;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.factorization.impl.ALS;
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

  public void setWeak(final VecOptimization weak) {
    this.weak = weak;
  }

  public void setLocal(final String localName) {
    this.localName = localName;
  }

  @Override
  public GradFacMulticlass create() {
    if (weak == null) {
      weak = defaultWeakBuilder.create();
    }
    return new GradFacMulticlass(weak, new ALS(alsIters), (Class<? extends L2>) DataTools.targetByName(localName));
  }}
