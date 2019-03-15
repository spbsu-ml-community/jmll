package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

import java.util.HashMap;
import java.util.Map;
import java.lang.Math;

public final class UserLambda {
    private final ArrayVec userEmbedding;
    private final Map<String, ArrayVec> itemEmbeddings;
    private final double beta;
    private final double otherItemsImportance;
    private double currentTime;
    private final Map<String, Double> lastTimeOfItems;
    private double commonSum;
    private final Map<String, Double> additionalSumByItems;
    private final ArrayVec commonUserDerivative;
    private final Map<String, ArrayVec> userDerivativeByItems;
    private final ArrayVec commonItemsDerivative;
    private final Map<String, ArrayVec> itemsDerivativeByItems;
    private final int dim;

    public UserLambda(ArrayVec userEmbedding, Map<String, ArrayVec> ItemsEmbeddings, double beta, double otherItemsImportance) {
        this.userEmbedding = userEmbedding;
        this.itemEmbeddings = ItemsEmbeddings;
        this.beta = beta;
        this.otherItemsImportance = otherItemsImportance;
        dim = userEmbedding.dim();

        currentTime = 0.;
        lastTimeOfItems = new HashMap<>();

        commonSum = 0;
        additionalSumByItems = new HashMap<>();

        commonUserDerivative = new ArrayVec(dim);
        VecTools.fill(commonUserDerivative, 0.);
        userDerivativeByItems = new HashMap<>();

        commonItemsDerivative = new ArrayVec(dim);
        VecTools.fill(commonItemsDerivative, 0);
        itemsDerivativeByItems = new HashMap<>();
    }

    public final void update(String ItemsId, double timeDelta) {
        timeDelta = 1.;
        if (!lastTimeOfItems.containsKey(ItemsId)) {
            lastTimeOfItems.put(ItemsId, currentTime);
        }
        double e = Math.exp(-beta * timeDelta);
        double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(ItemsId)));
        double scalarProduct = VecTools.multiply(userEmbedding, itemEmbeddings.get(ItemsId));

        if (!additionalSumByItems.containsKey(ItemsId)) {
            additionalSumByItems.put(ItemsId, 0.);
            ArrayVec ItemsSpecificItemsDerivative = new ArrayVec(dim);
            VecTools.fill(ItemsSpecificItemsDerivative, 0.);
            userDerivativeByItems.put(ItemsId, ItemsSpecificItemsDerivative);
            ArrayVec ItemsSpecificUserDerivative = new ArrayVec(dim);
            VecTools.fill(ItemsSpecificUserDerivative, 0.);
            itemsDerivativeByItems.put(ItemsId, ItemsSpecificUserDerivative);
        }

        // Updating lambda
        commonSum = e * commonSum + otherItemsImportance * scalarProduct;
        additionalSumByItems.put(ItemsId,
                decay * additionalSumByItems.get(ItemsId) + (1 - otherItemsImportance) * scalarProduct);

        // Updating user derivative
        ArrayVec commonUserDerivativeAdd = new ArrayVec();
        commonUserDerivativeAdd.assign(itemEmbeddings.get(ItemsId));
        commonUserDerivativeAdd.scale(otherItemsImportance);
        commonUserDerivative.scale(e);
        commonUserDerivative.add(commonUserDerivativeAdd);
        ArrayVec ItemsUserDerivativeAdd = new ArrayVec();
        ItemsUserDerivativeAdd.assign(itemEmbeddings.get(ItemsId));
        ItemsUserDerivativeAdd.scale(1 - otherItemsImportance);
        userDerivativeByItems.get(ItemsId).scale(decay);
        userDerivativeByItems.get(ItemsId).add(ItemsUserDerivativeAdd);

        // Updating Items derivative
        ArrayVec commonItemsDerivativeAdd = new ArrayVec(dim);
        commonItemsDerivativeAdd.assign(userEmbedding);
        commonItemsDerivativeAdd.scale(otherItemsImportance);
        commonItemsDerivative.scale(e);
        commonItemsDerivative.add(commonItemsDerivativeAdd);
        ArrayVec ItemsDerivativeByItemsAdd = new ArrayVec();
        ItemsDerivativeByItemsAdd.assign(userEmbedding);
        ItemsDerivativeByItemsAdd.scale(1 - otherItemsImportance);
        itemsDerivativeByItems.get(ItemsId).scale(decay);
        itemsDerivativeByItems.get(ItemsId).add(ItemsDerivativeByItemsAdd);

        lastTimeOfItems.put(ItemsId, currentTime);
        currentTime += timeDelta;
    }

    public final double getLambda(String ItemsId) {
        if (!additionalSumByItems.containsKey(ItemsId)) {
            return commonSum + VecTools.multiply(userEmbedding, itemEmbeddings.get(ItemsId));
        }
        return commonSum + VecTools.multiply(userEmbedding, itemEmbeddings.get(ItemsId)) +
                additionalSumByItems.get(ItemsId);
    }

    public final ArrayVec getLambdaUserDerivative(String ItemsId) {
        ArrayVec completeDerivative = new ArrayVec(dim);
        completeDerivative.assign(commonUserDerivative);
        completeDerivative.add(itemEmbeddings.get(ItemsId));
        if (userDerivativeByItems.containsKey(ItemsId)) {
            ArrayVec completeDerivativeAdd = new ArrayVec(dim);
            completeDerivativeAdd.assign(userDerivativeByItems.get(ItemsId));
            double decay = currentTime - lastTimeOfItems.get(ItemsId);
            completeDerivative.scale(decay);
            completeDerivative.add(completeDerivativeAdd);
        }
        return completeDerivative;
    }

    public final Map<String, ArrayVec> getLambdaItemsDerivative(String ItemsId) {
        Map<String, ArrayVec> derivative = new HashMap<>();
        for (String p : lastTimeOfItems.keySet()) {
            ArrayVec initialDerivative = new ArrayVec(dim);
            initialDerivative.assign(commonItemsDerivative);
            derivative.put(p, initialDerivative);
        }
        if (itemsDerivativeByItems.containsKey(ItemsId)) {
            double decay = currentTime - lastTimeOfItems.get(ItemsId);
            ArrayVec derivativeAdd = new ArrayVec(dim);
            derivativeAdd.assign(itemsDerivativeByItems.get(ItemsId));
            derivativeAdd.scale(decay);
            derivativeAdd.add(userEmbedding);
            derivative.get(ItemsId).add(derivativeAdd);
        }
        return derivative;
    }
}
