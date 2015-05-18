package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.loss.WeightedLoss;
import com.spbsu.ml.methods.trees.GreedyObliviousPolynomialTree;
import com.spbsu.ml.models.PolynomialObliviousTree;
import com.spbsu.ml.testUtils.FakePool;

import java.io.IOException;

/**
 * Created by towelenee on 5/11/15.
 * Tests for greedy polynomial Tree must have regression test and boosting test
 */
public class GreedyObliviousPolynomialTreeTest extends MethodsTests {
  public void testPOTBoost() throws IOException, InterruptedException {
    testWithBoosting(
        new GreedyObliviousPolynomialTree(GridTools.medianGrid(learn.vecData(), 32), 6, 2, 0.1),
        learn,
        validate,
        2000,
        0.005,
        MethodsTests.OUTPUT_SCORE | MethodsTests.OUTPUT_DRAW
    );
  }
  public void testOnSmallDataset() {
    Mx data = new VecBasedMx(4, 1);
    data.set(0,0);
    data.set(1,0.25);
    data.set(2, 0.5);
    data.set(3, 0.75);
    Vec target = new ArrayVec( 0, 1, 100, 101);
    int[] weights = new int[] {1, 1, 1, 1};
    final FakePool fakePool = new FakePool(data, target);
    final GreedyObliviousPolynomialTree polynomialTree
        = new GreedyObliviousPolynomialTree(GridTools.medianGrid(fakePool.vecData(), 32), 1, 1, 0);
    final PolynomialObliviousTree tree = polynomialTree.fit(
        fakePool.vecData(),
        new WeightedLoss<>(new L2(target, fakePool.vecData()), weights)
    );
    for (int i = 0; i < 4; i++) {
      assertEquals(target.get(i), tree.value(data.row(i)), 1e-10);
    }
  }

}
