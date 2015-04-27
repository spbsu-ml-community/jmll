package com.spbsu.exp;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.MethodsTests;
import com.spbsu.ml.loss.SatL2;
import com.spbsu.ml.methods.BootstrapOptimization;
import com.spbsu.ml.methods.GradientBoosting;
import com.spbsu.robustObliviousTree;

/**
 * Created by towelenee on 4/1/15.
 */
public class robustObliviousTreeTest extends MethodsTests {
    public void testRobustOTBoost() {
        final GradientBoosting<SatL2> boosting = new GradientBoosting<SatL2>(new BootstrapOptimization(new robustObliviousTree(GridTools.medianGrid(learn.vecData(), 32), 6), new FastRandom(0)), 2000, 0.01);
        new addBoostingListeners<SatL2>(boosting, learn.target(SatL2.class), learn, validate);
    }

}
