package com.spbsu.bernulli;

/**
 * User: Noxoomo
 * Date: 20.03.15
 * Time: 21:18
 */

public class FittedModel<Model> {
  public final double likelihood;
  public final Model model;

  public FittedModel(double likelihood, Model model) {
    this.likelihood = likelihood;
    this.model = model;
  }
}
