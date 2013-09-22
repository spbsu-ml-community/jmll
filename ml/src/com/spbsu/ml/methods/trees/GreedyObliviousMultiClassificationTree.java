package com.spbsu.ml.methods.trees;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.Model;
import com.spbsu.ml.MultiClassModel;
import com.spbsu.ml.Oracle1;
import com.spbsu.ml.data.DataSet;
import com.spbsu.ml.data.DataTools;
import com.spbsu.ml.data.impl.BinarizedDataSet;
import com.spbsu.ml.data.impl.Bootstrap;
import com.spbsu.ml.loss.LogLikelihoodSigmoid;
import com.spbsu.ml.methods.MLMultiClassMethodOrder1;
import com.spbsu.ml.models.ObliviousMultiClassTree;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * User: solar
 * Date: 30.11.12
 * Time: 17:01
 */
public class GreedyObliviousMultiClassificationTree implements MLMultiClassMethodOrder1 {
    private final Random rng;
    private final BinarizedDataSet ds;
    private final int depth;
    private BFGrid grid;

    public GreedyObliviousMultiClassificationTree(Random rng, DataSet ds, BFGrid grid, int depth) {
        this.rng = rng;
        this.depth = depth;
        this.ds = new BinarizedDataSet(ds, grid);
        this.grid = grid;
    }

    @Override
    public MultiClassModel fit(DataSet learn, Oracle1 loss) {
        final Vec[] points = new Vec[DataTools.countClasses(learn.target())];
        for (int i = 0; i < points.length; i++) {
            points[i] = new ArrayVec(learn.power());
        }
        return fit(learn, loss, points);
    }

    @Override
    public Model fit(DataSet learn, Oracle1 loss, Vec start) {
        throw new UnsupportedOperationException("For multiclass methods continuation from fixed point is not supported");
    }

    public ObliviousMultiClassTree fit(DataSet ds, Oracle1 loss, Vec[] point) {
        if (!(loss instanceof LogLikelihoodSigmoid))
            throw new IllegalArgumentException("Log likelihood with sigmoid value function supported only");
        final Vec target = ds instanceof Bootstrap ? ((Bootstrap) ds).original().target() : ds.target();
        final List<BFGrid.BinaryFeature> conditions = new ArrayList<BFGrid.BinaryFeature>(depth);
        MultiLLClassificationLeaf seed = new MultiLLClassificationLeaf(this.ds, point, target, VecTools.fill(new ArrayVec(ds.power()), 1.));
        MultiLLClassificationLeaf[] currentLayer = new MultiLLClassificationLeaf[]{seed};
        for (int level = 0; level < depth; level++) {
            double[] scores = new double[grid.size()];
            for (MultiLLClassificationLeaf leaf : currentLayer) {
                leaf.score(scores);
            }
            final int max = ArrayTools.max(scores);
            if (max < 0)
                throw new RuntimeException("Can not find optimal split!");
            BFGrid.BinaryFeature bestSplit = grid.bf(max);
            final MultiLLClassificationLeaf[] nextLayer = new MultiLLClassificationLeaf[1 << (level + 1)];
            for (int l = 0; l < currentLayer.length; l++) {
                MultiLLClassificationLeaf leaf = currentLayer[l];
                nextLayer[2 * l] = leaf;
                nextLayer[2 * l + 1] = leaf.split(bestSplit);
            }
            conditions.add(bestSplit);
            currentLayer = nextLayer;
        }
        final MultiLLClassificationLeaf[] leaves = currentLayer;

        double[] values = new double[leaves.length];
        double[] weights = new double[leaves.length];
        boolean[][] masks = new boolean[leaves.length][];

        {
            for (int i = 0; i < weights.length; i++) {
                values[i] = leaves[i].alpha();
                masks[i] = leaves[i].mask();
                weights[i] = leaves[i].size();
            }
        }
        return new ObliviousMultiClassTree(conditions, values, weights, masks);
    }
}
