package com.spbsu.exp.dl.functions;

import com.spbsu.exp.dl.functions.unary.HeavisideFA;
import com.spbsu.exp.dl.functions.unary.IdenticalFA;
import com.spbsu.exp.dl.functions.unary.SigmoidFA;
import com.spbsu.exp.dl.functions.unary.TanhFA;

/**
 * jmll
 * ksen
 * 13.December.2014 at 14:01
 */
public enum FunctionType {

  HEAVISIDE(){
    HeavisideFA getInstance() {
      return new HeavisideFA();
    }
  },
  IDENTICAL(){
    IdenticalFA getInstance() {
      return new IdenticalFA();
    }
  },
  SIGMOID(){
    SigmoidFA getInstance() {
      return new SigmoidFA();
    }
  },
  TANH(){
    TanhFA getInstance() {
      return new TanhFA();
    }
  }
  ;

  abstract ArrayUnaryFunction getInstance();

}
