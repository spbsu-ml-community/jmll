package com.spbsu.ml.methods.cart;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid;

import java.util.List;

/**
 * Created by n_buga on 10.03.17.
 */
public class CARTTree extends Func.Stub {
    private final List<LeafConditions> features;
    private final double[] values;
    private final double[] basedOn;
    private final BFGrid grid;

    public CARTTree(final List<LeafConditions> features, final double[] values, final double[] basedOn) {
        grid = features.get(0).getLeafConditions()[0].row().grid();
        this.basedOn = basedOn;
        this.features = features;
        this.values = values;
    }

    @Override
    public double value(Vec x) {
        for (int i = 0; i < features.size(); i++) {
            if (features.get(i).isMatch(x)) {
                return values[i];
            }
        }
        throw new IllegalStateException("There're no leaf for x matching conditions.");
    }

    @Override
    public int dim() {
        return grid.rows();
    }
}
