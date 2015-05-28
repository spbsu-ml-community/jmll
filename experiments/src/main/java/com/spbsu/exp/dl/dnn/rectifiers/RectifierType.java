package com.spbsu.exp.dl.dnn.rectifiers;

/**
 * jmll
 *
 * @author ksenon
 */
public enum  RectifierType {

  SIGMOID {
    @Override
    public Rectifier getInstance() {
      return new Sigmoid();
    }
  },
  FLAT {
    @Override
    public Rectifier getInstance() {
      return new Flat();
    }
  },
  TANH {
    @Override
    public Rectifier getInstance() {
      return new Tanh();
    }
  },
  BIPOLAR_SIGM {
    @Override
    public Rectifier getInstance() {
      return new BipolarSigmoid();
    }
  }
  ;

  public abstract Rectifier getInstance();

}
