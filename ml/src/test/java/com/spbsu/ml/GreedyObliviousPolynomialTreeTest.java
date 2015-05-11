package com.spbsu.ml;

import com.spbsu.ml.methods.trees.GreedyObliviousPolynomialTree;

import java.io.IOException;

/**
 * Created by towelenee on 5/11/15.
 */
public class GreedyObliviousPolynomialTreeTest extends MethodsTests {
  public void testGet() {
    final int dim = 14;
    final int maxElement = 5;
    final int count = GreedyObliviousPolynomialTree.count(maxElement, dim);
    for (int i = 0; i < count; i++) {
    }
  }

  public void testPOTBoost() throws IOException, InterruptedException {
    testWithBoosting(
        new GreedyObliviousPolynomialTree(GridTools.medianGrid(learn.vecData(), 32), 6, 2),
        learn,
        validate,
        2000,
        0.02,
        MethodsTests.OUTPUT_SCORE | MethodsTests.OUTPUT_DRAW
    );
  }

}
