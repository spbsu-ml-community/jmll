package com.spbsu.exp.cart;

/**
 * Created by n_buga on 11.03.17.
 */

import com.spbsu.commons.math.Trans;
import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.VecBuilder;
import com.spbsu.commons.util.Pair;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.GridTools;
import com.spbsu.ml.data.set.VecDataSet;
import com.spbsu.ml.data.set.impl.VecDataSetImpl;
import com.spbsu.ml.loss.L2;
import com.spbsu.ml.methods.cart.CARTTree;
import com.spbsu.ml.methods.cart.CARTTreeOptimization;
import com.spbsu.ml.methods.trees.GreedyObliviousTree;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestCartTreeOptimization {

    private Pair<VecDataSet, Vec> prepareData() {
        VecDataSet dataSet;
        VecBuilder builderData = new VecBuilder();
        builderData.append(0);
        builderData.append(1);
        builderData.append(2);
        builderData.append(3);
        VecBuilder builderTarget = new VecBuilder();
        builderTarget.append(0);
        builderTarget.append(1);
        builderTarget.append(2);
        builderTarget.append(3);
        Mx dataMx = new VecBasedMx(1, builderData.build());
        dataSet = new VecDataSetImpl(dataMx, null);
        return new Pair<>(dataSet, builderTarget.build());
    }

    @Test
    public void testOutcome2Points() {
        final int depth = 3;
        final int binFactor = 1;
        Pair<VecDataSet, Vec> data = prepareData();
        BFGrid bfData = GridTools.medianGrid(data.getFirst(), binFactor);
        CARTTreeOptimization opt = new CARTTreeOptimization(bfData, depth);
        CARTTree tree = opt.fit(data.getFirst(), new L2(data.getSecond(), data.getFirst()));
 //       System.out.print(tree.dim());
    }

    @Test
    public void testCompare2PointsOblivious() {
        final int depth = 3;
        final int binFactor = 1;
        Pair<VecDataSet, Vec> data = prepareData();
        BFGrid bfData = GridTools.medianGrid(data.getFirst(), binFactor);
        GreedyObliviousTree opt = new GreedyObliviousTree(bfData, depth);
        Trans tree = opt.fit(data.getFirst(), new L2(data.getSecond(), data.getFirst()));
    }
}
