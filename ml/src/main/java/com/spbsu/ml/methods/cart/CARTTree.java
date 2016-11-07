package com.spbsu.ml.methods.cart;

import com.spbsu.commons.math.Func;
import com.spbsu.commons.math.vectors.Vec;

import java.util.List;

/**
 * Created by n_buga on 17.10.16.
 */
public class CARTTree extends Func.Stub {
    final private List<Leaf> leaves;

    public CARTTree(List<Leaf> leaves) {
        this.leaves = leaves;
    }

    public double value(Vec x) {
        for (Leaf leaf: leaves) {
            if (leaf.getListFeatures().check(x)) {
                return leaf.getValue();
            }
        }
        return 0;
    }

    public int dim() {
        return leaves.size();
    }
}
