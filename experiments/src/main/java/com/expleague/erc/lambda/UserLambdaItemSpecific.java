package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import gnu.trove.iterator.TIntDoubleIterator;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.lang.Math;

public final class UserLambdaItemSpecific implements UserLambda {
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

    public UserLambdaItemSpecific(final Vec userEmbedding, final TIntObjectMap<Vec> itemsEmbeddings, final double beta,
                                  final double otherItemsImportance) {
        this.userEmbedding = userEmbedding;
        this.itemEmbeddings = itemsEmbeddings;
        this.beta = beta;
        this.otherItemsImportance = otherItemsImportance;

        currentTime = 0.;
        lastTimeOfItems = new TIntDoubleHashMap();

        commonSum = 0;
        additionalSumByItems = new TIntDoubleHashMap();

        commonUserDerivative = new ArrayVec(userEmbedding.dim());
        VecTools.fill(commonUserDerivative, 0.);
        userDerivativeByItems = new TIntObjectHashMap<>();

        commonItemsDerivative = new ArrayVec(userEmbedding.dim());
        VecTools.fill(commonItemsDerivative, 0);
        itemsDerivativeByItems = new TIntObjectHashMap<>();
    }

    @Override
    public void reset() {
        currentTime = 0.;
        lastTimeOfItems.clear();
        commonSum = 0.;
        additionalSumByItems.clear();
        VecTools.fill(commonUserDerivative, 0.);
        userDerivativeByItems.clear();
        VecTools.fill(commonItemsDerivative, 0);
        itemsDerivativeByItems.clear();
    }

    @Override
    public final void update(final int itemId, double timeDelta) {
        timeDelta = 1.;
        if (!lastTimeOfItems.containsKey(itemId)) {
            lastTimeOfItems.put(itemId, currentTime);
            additionalSumByItems.put(itemId, 0.);
            Vec itemsSpecificItemsDerivative = new ArrayVec(userEmbedding.dim());
            VecTools.fill(itemsSpecificItemsDerivative, 0.);
            userDerivativeByItems.put(itemId, itemsSpecificItemsDerivative);
            Vec itemsSpecificUserDerivative = new ArrayVec(userEmbedding.dim());
            VecTools.fill(itemsSpecificUserDerivative, 0.);
            itemsDerivativeByItems.put(itemId, itemsSpecificUserDerivative);
        }
        final double e = Math.exp(-beta * timeDelta);
        final double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(itemId)));
        final double interactionEffect = VecTools.multiply(userEmbedding, itemEmbeddings.get(itemId));

        // Updating lambda
        commonSum = e * commonSum + otherItemsImportance * interactionEffect;
        additionalSumByItems.put(itemId,
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

    @Override
    public final double getLambda(final int itemId) {
        double baseLambda = commonSum + VecTools.multiply(userEmbedding, itemEmbeddings.get(itemId));
        if (!additionalSumByItems.containsKey(itemId)) {
            return baseLambda;
        }
        return baseLambda + additionalSumByItems.get(itemId);
    }

    @Override
    public final Vec getLambdaUserDerivative(final int itemId) {
        final Vec completeDerivative = VecTools.copy(commonUserDerivative);
        VecTools.append(completeDerivative, itemEmbeddings.get(itemId));
        if (userDerivativeByItems.containsKey(itemId)) {
            final Vec itemSpecificDerivative = VecTools.copy(userDerivativeByItems.get(itemId));
            final double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(itemId)));
            VecTools.scale(completeDerivative, decay);
            VecTools.append(completeDerivative, itemSpecificDerivative);
        }
        return completeDerivative;
    }

    @Override
    public final TIntObjectMap<Vec> getLambdaItemsDerivative(final int itemId) {
        final TIntObjectMap<Vec> derivative = new TIntObjectHashMap<>();
        for(TIntDoubleIterator it = lastTimeOfItems.iterator(); it.hasNext();) {
            it.advance();
            derivative.put(it.key(), VecTools.copy(commonItemsDerivative));
        }
        if (itemsDerivativeByItems.containsKey(itemId)) {
            final double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(itemId)));
            final Vec itemSpecificDerivative = VecTools.copy(itemsDerivativeByItems.get(itemId));
            VecTools.scale(itemSpecificDerivative, decay);
            VecTools.append(itemSpecificDerivative, userEmbedding);
            VecTools.append(derivative.get(itemId), itemSpecificDerivative);
        }
        return derivative;
    }
}
