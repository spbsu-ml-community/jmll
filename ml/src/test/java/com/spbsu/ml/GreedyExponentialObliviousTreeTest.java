package com.spbsu.ml;

import com.spbsu.ml.methods.trees.GreedyExponentialObliviousTree;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Test for exponential probability idea
 * Created by towelenee on 5/9/15.
 */
public class GreedyExponentialObliviousTreeTest extends MethodsTests {
  public void testUpperBound() {
    List<Double> list = Arrays.asList(0., 1., 1., 2., 3., 3., 3.);
    assertEquals(GreedyExponentialObliviousTree.upperBound(list, 2.0), 4);
    assertEquals(GreedyExponentialObliviousTree.upperBound(list, 1.0), 3);
    assertEquals(GreedyExponentialObliviousTree.upperBound(list, 0.0), 1);
    assertEquals(GreedyExponentialObliviousTree.upperBound(list, 3.0), 7);

    assertEquals(GreedyExponentialObliviousTree.upperBound(list, 100.0), list.size());
    assertEquals(GreedyExponentialObliviousTree.upperBound(list, -1.), 0);
  }

  public void testLowerBound() {
    List<Double> list = Arrays.asList(0., 1., 1., 2., 3., 3., 3.);
    assertEquals(3, GreedyExponentialObliviousTree.lowerBound(list, 2.0));
    assertEquals(1, GreedyExponentialObliviousTree.lowerBound(list, 1.0));
    assertEquals(0, GreedyExponentialObliviousTree.lowerBound(list, 0.0));
    assertEquals(4, GreedyExponentialObliviousTree.lowerBound(list, 3.0));

    assertEquals(GreedyExponentialObliviousTree.lowerBound(list, 100.0), list.size());
    assertEquals(GreedyExponentialObliviousTree.lowerBound(list, -1.), 0);
  }
/*
  public void testProbabilityOfFit() {
    List<Double> list = Arrays.asList(0., 1., 1., 2., 3., 5., 5., 8., 8., 8., 9.);
    assertEquals(1 - 1e-5, GreedyExponentialObliviousTree.getProbabilityOfFit(list, 0., 3., 0.1), 1e-15);

    assertEquals(1 - 1e-4, GreedyExponentialObliviousTree.getProbabilityOfFit(list, 2., 5., 0.1), 1e-15);

    assertEquals(1e-4, GreedyExponentialObliviousTree.getProbabilityOfFit(list, 3., 0., 0.1), 1e-15);

    assertEquals(1e-2, GreedyExponentialObliviousTree.getProbabilityOfFit(list, 3., 1., 0.1), 1e-15);

    {
      double p = (1e-3 + 1e-2) / 2;
      assertEquals(1 - p, GreedyExponentialObliviousTree.getProbabilityOfFit(list, 1., 2., 0.1), 1e-15);
    }

    {
      double p = (1e-1 + 1e-2 + 1e-3) / 3;
      assertEquals(p, GreedyExponentialObliviousTree.getProbabilityOfFit(list, 8., 5., 0.1), 1e-15);
    }
  }
*/
  public void testTs() throws FileNotFoundException {
    new GreedyExponentialObliviousTree(GridTools.medianGrid(learn.vecData(), 32), learn.vecData(), 6, 1 - Math.pow(2, -8)).output();
  }
  public void testEOTBoost() throws IOException, InterruptedException {
    testWithBoosting(
        new GreedyExponentialObliviousTree(GridTools.medianGrid(learn.vecData(), 32), learn.vecData(), 6, 1 - Math.pow(2, -5)),
        learn,
        validate,
        2000,
        0.02,
        OUTPUT_SCORE | OUTPUT_DRAW,
        "graph.tsv");
  }
  public void testEOTBoostMulti() throws IOException, InterruptedException {
    for (int regulation = 0; regulation < 5; regulation++)
    {
      for(double step = 0.001; step < 0.011; step += 0.001) {
        testWithBoosting(
            new GreedyExponentialObliviousTree(GridTools.medianGrid(learn.vecData(), 32), learn.vecData(), 6, 1 - Math.pow(2, -regulation)),
            learn,
            validate,
            1000,
            step,
            OUTPUT_SCORE | OUTPUT_DRAW,
            "exp-" + regulation + "step-" + step
        );
      }
    }
  }


}
