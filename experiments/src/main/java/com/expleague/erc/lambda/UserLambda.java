package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.io.Serializable;
import java.lang.Math;
import java.util.function.DoubleUnaryOperator;

public final class UserLambda {
    private final Vec userEmbedding;
    private final TIntObjectMap<Vec> itemEmbeddings;
    private final double beta;
    private final double otherItemsImportance;
    private double currentTime;
    private final TIntDoubleMap lastTimeOfItems;
    private double commonSum;
    private final TIntDoubleMap additionalSumByItems;
    private final Vec commonUserDerivative;
    private final TIntObjectMap<Vec> userDerivativeByItems;
    private final Vec commonItemsDerivative;
    private final TIntObjectMap<Vec> itemsDerivativeByItems;
    private final Vec zeroVec;

    public UserLambda(final Vec userEmbedding, final TIntObjectMap<Vec> itemsEmbeddings, final double beta,
                      final double otherItemsImportance) {
        this.userEmbedding = userEmbedding;
        this.itemEmbeddings = itemsEmbeddings;
        this.beta = beta;
        this.otherItemsImportance = otherItemsImportance;
        zeroVec = new ArrayVec(userEmbedding.dim());
        VecTools.fill(zeroVec, 0.);

        currentTime = 0.;
        lastTimeOfItems = new TIntDoubleHashMap();

        commonSum = 0;
        additionalSumByItems = new TIntDoubleHashMap();

        commonUserDerivative = VecTools.copy(zeroVec);
        VecTools.fill(commonUserDerivative, 0.);
        userDerivativeByItems = new TIntObjectHashMap<>();

        commonItemsDerivative = VecTools.copy(zeroVec);
        VecTools.fill(commonItemsDerivative, 0);
        itemsDerivativeByItems = new TIntObjectHashMap<>();
    }

    public final void update(final int itemId, double timeDelta) {
        timeDelta = 1.;
        if (!lastTimeOfItems.containsKey(itemId)) {
            lastTimeOfItems.put(itemId, currentTime);
            additionalSumByItems.put(itemId, 0.);
            Vec itemsSpecificItemsDerivative = VecTools.copy(zeroVec);
            VecTools.fill(itemsSpecificItemsDerivative, 0.);
            userDerivativeByItems.put(itemId, itemsSpecificItemsDerivative);
            Vec itemsSpecificUserDerivative = VecTools.copy(zeroVec);
            VecTools.fill(itemsSpecificUserDerivative, 0.);
            itemsDerivativeByItems.put(itemId, itemsSpecificUserDerivative);
        }
        final double e = Math.exp(-beta * timeDelta);
        final double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(itemId)));
        final double interactionEffect = VecTools.multiply(userEmbedding, itemEmbeddings.get(itemId));

        // Updating lambda
//        commonSum = commonSum + otherItemsImportance * interactionEffect;
        commonSum = e * commonSum + otherItemsImportance * interactionEffect;
        additionalSumByItems.put(itemId,
//                additionalSumByItems.get(itemId) + (1 - otherItemsImportance) * interactionEffect);
                decay * additionalSumByItems.get(itemId) + (1 - otherItemsImportance) * interactionEffect);

        // Updating user derivative
        Vec commonUserDerivativeAdd = VecTools.copy(itemEmbeddings.get(itemId));
        VecTools.scale(commonUserDerivativeAdd, otherItemsImportance);
        VecTools.scale(commonUserDerivative, e);
        VecTools.append(commonUserDerivative, commonUserDerivativeAdd);
        Vec itemsUserDerivativeAdd = VecTools.copy(itemEmbeddings.get(itemId));
        VecTools.scale(itemsUserDerivativeAdd, 1 - otherItemsImportance);
        VecTools.scale(userDerivativeByItems.get(itemId), decay);
        VecTools.append(userDerivativeByItems.get(itemId), itemsUserDerivativeAdd);

        // Updating Items derivative
        Vec commonItemsDerivativeAdd = VecTools.copy(userEmbedding);
        VecTools.scale(commonItemsDerivativeAdd, otherItemsImportance);
        VecTools.scale(commonItemsDerivative, e);
        VecTools.append(commonItemsDerivative, commonItemsDerivativeAdd);
        Vec itemsDerivativeByItemsAdd = VecTools.copy(userEmbedding);
        VecTools.scale(itemsDerivativeByItemsAdd, 1 - otherItemsImportance);
        VecTools.scale(itemsDerivativeByItems.get(itemId), decay);
        VecTools.append(itemsDerivativeByItems.get(itemId), itemsDerivativeByItemsAdd);

        lastTimeOfItems.put(itemId, currentTime);
        currentTime += timeDelta;
    }

    public final double getLambda(final int itemId) {
        double baseLambda = commonSum + VecTools.multiply(userEmbedding, itemEmbeddings.get(itemId));
        if (!additionalSumByItems.containsKey(itemId)) {
            return baseLambda;
        }
        return baseLambda + additionalSumByItems.get(itemId);
    }

    public final Vec getLambdaUserDerivative(final int itemId) {
        Vec completeDerivative = VecTools.copy(commonUserDerivative);
        VecTools.append(completeDerivative, itemEmbeddings.get(itemId));
        if (userDerivativeByItems.containsKey(itemId)) {
            Vec completeDerivativeAdd = VecTools.copy(userDerivativeByItems.get(itemId));
            double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(itemId)));  // why now exp?
//            double decay = currentTime - lastTimeOfItems.get(itemId);  // why now exp?
            VecTools.scale(completeDerivative, decay);
            VecTools.append(completeDerivative, completeDerivativeAdd);
        }
        return completeDerivative;
    }

    public final TIntObjectMap<Vec> getLambdaItemsDerivative(final int itemId) {
        TIntObjectMap<Vec> derivative = new TIntObjectHashMap<>();
        for (int p : lastTimeOfItems.keys()) {
            Vec initialDerivative = VecTools.copy(commonItemsDerivative);
            derivative.put(p, initialDerivative);
        }
        if (itemsDerivativeByItems.containsKey(itemId)) {
            double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(itemId)));  // why now exp?
//            double decay = currentTime - lastTimeOfItems.get(itemId);  // why now exp?
            Vec derivativeAdd = VecTools.copy(itemsDerivativeByItems.get(itemId));
            VecTools.scale(derivativeAdd, decay);
            VecTools.append(derivativeAdd, userEmbedding);
            VecTools.append(derivative.get(itemId), derivativeAdd);
        }
        return derivative;
    }

    public static class IdentityTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return v;
        }
    }

    public static class IdentityDerivativeTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return 1;
        }
    }

    public static class AbsTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return Math.abs(v);
        }
    }

    public static class AbsDerivativeTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return Math.signum(v);
        }
    }

    public static class SqrTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return v * v;
        }
    }

    public static class SqrDerivativeTransform implements DoubleUnaryOperator, Serializable {
        @Override
        public double applyAsDouble(double v) {
            return 2 * v;
        }
    }
}
