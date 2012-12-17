package com.spbsu.ml.loss;

/**
 * User: solar
 * Date: 23.12.2010
 * Time: 16:04:03
 */
public interface IterativeCalculator {
    void remove(int index);
    void add(int index);
    
    double value();
}
