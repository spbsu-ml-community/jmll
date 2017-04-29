package com.spbsu.ml.methods.cart;

import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.ml.BFGrid.BinaryFeature;

/**
 * Created by n_buga on 11.03.17.
 */
public class LeafConditions {

    final private BinaryFeature leafConditions[];
    final private boolean answers[];

    public LeafConditions() {
        leafConditions = new BinaryFeature[0];
        answers = new boolean[0];
    }

    public LeafConditions(LeafConditions parentLeafConditions, BinaryFeature newCondition, boolean ans) {
        int newSize = parentLeafConditions.getLeafConditions().length + 1;
        leafConditions = new BinaryFeature[newSize];
        answers = new boolean[newSize];
        for (int i = 0; i < newSize - 1; i++) {
            leafConditions[i] = parentLeafConditions.getLeafConditions()[i];
            answers[i] = parentLeafConditions.getAnswers()[i];
        }
        leafConditions[newSize - 1] = newCondition;
        answers[newSize - 1] = ans;
    }

    public BinaryFeature[] getLeafConditions() {
        return leafConditions;
    }

    public boolean[] getAnswers() {
        return answers;
    }

    public boolean isMatch(Vec x) {
        for (int i = 0; i < leafConditions.length; i++) {
            if (leafConditions[i].value(x) != answers[i]) {
                return false;
            }
        }
        return true;
    }
}
