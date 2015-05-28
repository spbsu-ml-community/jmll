package com.spbsu.exp.dl.dnn.config;

import com.spbsu.exp.dl.dnn.InitMethod;

/**
 * jmll
 *
 * @author ksenon
 */
public class SolverConfig {

  public int batchSize;
  public int epochsNumber;

  public double learningRate;

  public InitMethod initMethod;

  public NetConfig net;

  public boolean debug;

}
