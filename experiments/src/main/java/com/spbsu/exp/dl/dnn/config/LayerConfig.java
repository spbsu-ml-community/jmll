package com.spbsu.exp.dl.dnn.config;

import com.spbsu.exp.dl.dnn.rectifiers.RectifierType;

/**
 * jmll
 *
 * @author ksenon
 */
public class LayerConfig {

  public int inputSize;
  public int outputSize;
  public RectifierType rectifierType;

  public double bias            = 0;
  public double bias_b          = 0;
  public double dropoutFraction = 0;

  public boolean isTrain        = true;

}
