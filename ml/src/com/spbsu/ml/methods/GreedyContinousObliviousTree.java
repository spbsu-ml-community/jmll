package com.spbsu.ml.methods;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.Histogram;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.loss.L2Loss;
import com.spbsu.ml.models.ObliviousTree;
import com.spbsu.ml.optimization.ConvexFunction;
import com.spbsu.ml.optimization.QuadraticFunction;
import gnu.trove.TDoubleDoubleProcedure;
import gnu.trove.TIntArrayList;
import com.spbsu.ml.optimization.QuadraticFunction;
import com.spbsu.ml.models.ContinousObliviousTree;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: towelenee
 * Date: 14.05.13
 * Time: 21:09
 * To change this template use File | Settings | File Templates.
 */
public class GreedyContinousObliviousTree extends GreedyTDRegion {
    private final int depth;
    private final GreedyObliviousTree nonContinousVersion;

    public GreedyContinousObliviousTree(Random rng, DataSet ds, BFGrid grid, int depth) {
        super(rng, ds, grid, 1./3, 0);
        nonContinousVersion = new GreedyObliviousTree(rng, ds, grid, depth);
        this.depth = depth;
    }

    @Override
    public ContinousObliviousTree fit(DataSet ds, Oracle1 loss) {
        ObliviousTree x = nonContinousVersion.fit(ds, loss);
        //No continous for a while
        /*Mx mxA = new Mx;
        Vec w = new Vec();
        double w0 = 0;
        QuadraticFunction x= new QuadraticFunction(mxA,w,w0);*/
        List<BFGrid.BinaryFeature> features = x.getFeatures();
        double value[][] = new double[1 << features.size()][(features.size() + 1) * (features.size() + 1)];
        /*for(int i = 0; i < features.size();i++){
            double c = features.get(i).condition;
        } */
        for(int i = 0;i < ds.data().columns(); i++){
            value[x.bin(ds.data().col(i))][features.size() * (features.size() + 1)] = ds.target().get(i);
        }

        return new ContinousObliviousTree(features,value);
    }


    private class BestBFFinder implements TDoubleDoubleProcedure {
        double score = 0;
        int fold = 0;
        double bestScore = Double.MAX_VALUE;
        int bestFeature = -1;

        int bfIndex = 0;

        final double[] totals;
        final double[] totalWeights;
        final int complexity;

        BestBFFinder(double[] totals, double[] totalWeights, int complexity) {
            this.totals = totals;
            this.totalWeights = totalWeights;
            this.complexity = complexity;
        }

        @Override
        public boolean execute(double weight, double sum) {
            double rightScore = score(this.totalWeights[fold] - weight, totals[fold] - sum, complexity);
            double leftScore = score(weight, sum, complexity);
            score += rightScore + leftScore;
            fold++;
            return true;
        }

        public void advance() {
            if (bestScore > score) {
                bestScore = score;
                bestFeature = bfIndex;
            }
            fold = 0;
            score = 0;
            bfIndex++;
        }

        public int bestSplit() {
            return bestFeature;
        }
    }
}