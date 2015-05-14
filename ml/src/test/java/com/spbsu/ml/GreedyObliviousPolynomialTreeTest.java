package com.spbsu.ml;

import com.spbsu.ml.methods.trees.GreedyObliviousPolynomialTree;

import java.io.IOException;

/**
 * Created by towelenee on 5/11/15.
 * Tests for greedy polynomial Tree must have regression test and boosting test
 */
public class GreedyObliviousPolynomialTreeTest extends MethodsTests {
  public void testPOTBoost() throws IOException, InterruptedException {
    testWithBoosting(
        new GreedyObliviousPolynomialTree(GridTools.medianGrid(learn.vecData(), 32), 6, 1, 1000),
        learn,
        validate,
        2000,
        0.02,
        MethodsTests.OUTPUT_SCORE | MethodsTests.OUTPUT_DRAW
    );
  }

}
