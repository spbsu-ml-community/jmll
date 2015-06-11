package com.spbsu.ml;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.tools.DataTools;
import com.spbsu.ml.data.tools.FeaturesTxtPool;
import com.spbsu.ml.methods.trees.GreedyExponentialObliviousTree;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import com.spbsu.ml.methods.trees.PacGreedyPolynomialObliviousTree;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by towelenee on 5/26/15.
 */
public class LetorTests extends MethodsTests {
  protected FeaturesTxtPool test;
  @Override
  protected void setUp() throws IOException {
    learn = DataTools.loadLetorFile("file", new FileReader("/home/towelenee/pool/Fold1/train.txt"));
    validate = DataTools.loadLetorFile("file", new FileReader("/home/towelenee/pool/Fold1/vali.txt"));
    test = DataTools.loadLetorFile("file", new FileReader("/home/towelenee/pool/Fold1/test.txt"));
    rng = new FastRandom(0);
  }
  public void testOTFinal() throws IOException, InterruptedException {
    for (int attempt = 0; attempt < 100; attempt++)
    {
      testWithBoosting(
          new GreedyObliviousTree(GridTools.medianGrid(learn.vecData(), 32), 6),
          learn,
          test,
          2000,
          0.002,
          OUTPUT_SCORE | OUTPUT_DRAW,
          "ot//ot-final-" + attempt
      );
    }

  }

  public void testPACFinal() throws IOException, InterruptedException {
    for (int attempt = 0; attempt < 100; attempt++)
    {
      testWithBoosting(
          new PacGreedyPolynomialObliviousTree(GridTools.medianGrid(learn.vecData(), 32), 6, 2, 1e2),
          learn,
          test,
          1500,
          0.00125, // final
          MethodsTests.OUTPUT_SCORE | MethodsTests.OUTPUT_DRAW,
          "pac//pac-final-" + attempt
      );
    }

  }

  public void testEOTFinal() throws IOException, InterruptedException {
    for (int attempt = 0; attempt < 100; attempt++)
    {
      testWithBoosting(
          new GreedyExponentialObliviousTree(GridTools.medianGrid(learn.vecData(), 32), learn.vecData(), 6, 1 - Math.pow(2, -5)),
          learn,
          test,
          1500,
          0.002,
          MethodsTests.OUTPUT_SCORE | MethodsTests.OUTPUT_DRAW,
          "eot//eot-final-" + attempt
      );
      //}
    }

  }


}
