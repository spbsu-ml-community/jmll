package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class UserLambdaSingle implements UserLambda {
    private final Vec userEmbedding;
    private final TIntObjectMap<Vec> itemEmbeddings;
    private final double beta;
    private final int dim;
    private final double initialLambda;

    private double lambda;
    private double currentTime;
    private double lastTime;
    private final TIntDoubleMap lastTimeOfItems;
    private final Vec userDerivative;
    private final TIntObjectMap<Vec> itemDerivatives;

    public UserLambdaSingle(final Vec userEmbedding, final TIntObjectMap<Vec> itemsEmbeddings, final double beta,
                            final double initialValue) {
        this.userEmbedding = userEmbedding;
        this.itemEmbeddings = itemsEmbeddings;
        this.beta = beta;
        dim = userEmbedding.dim();

        currentTime = 0.;
        lastTime = 0.;
        lastTimeOfItems = new TIntDoubleHashMap();

        initialLambda = initialValue;
        lambda = initialLambda;
        userDerivative = new ArrayVec(dim);
        itemDerivatives = new TIntObjectHashMap<>();
    }

    @Override
    public void reset() {
        lambda = initialLambda;
        currentTime = 0.;
        lastTime = 0.;
        lastTimeOfItems.clear();
        itemDerivatives.clear();
        VecTools.fill(userDerivative, 0);
    }

    @Override
    public final void update(final int itemId, double timeDelta) {
        timeDelta = 1.;
        if (!lastTimeOfItems.containsKey(itemId)) {
            lastTimeOfItems.put(itemId, currentTime);
            itemDerivatives.put(itemId, new ArrayVec(dim));
        }

        // Updating lambda
        final double e = Math.exp(-beta * timeDelta);
        final double interactionEffect = VecTools.multiply(userEmbedding, itemEmbeddings.get(itemId));
        lambda = e * lambda + interactionEffect;

        // Updating user derivative
        final Vec commonUserDerivativeAdd = VecTools.copy(itemEmbeddings.get(itemId));
        VecTools.scale(userDerivative, e);
        VecTools.append(userDerivative, commonUserDerivativeAdd);

        // Updating item derivative
        final Vec itemDerivativeDelta = VecTools.copy(userEmbedding);
        final Vec itemDerivative = itemDerivatives.get(itemId);
        final double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(itemId)));
        VecTools.scale(itemDerivative, decay);
        VecTools.append(itemDerivative, itemDerivativeDelta);

        lastTimeOfItems.put(itemId, currentTime);
        currentTime += timeDelta;
    }

    @Override
    public final double getLambda(final int itemId) {
        return lambda;
    }

    public final double getLambda() {
        return lambda;
    }

    public final Vec getLambdaUserDerivative() {
        return VecTools.copy(userDerivative);
    }

    @Override
    public Vec getLambdaUserDerivative(int itemId) {
        throw new UnsupportedOperationException();
    }

    public final TIntObjectMap<Vec> getLambdaItemsDerivative() {
        final TIntObjectMap<Vec> derivative = new TIntObjectHashMap<>();
        itemDerivatives.forEachEntry((curItemId, itemDerivative) -> {
            derivative.put(curItemId, VecTools.copy(itemDerivative));
            return true;
        });
        return derivative;
    }

    @Override
    public TIntObjectMap<Vec> getLambdaItemsDerivative(int itemId) {
        throw new UnsupportedOperationException();
    }

    public static TIntDoubleMap makeUserLambdaInitialValues(List<Event> history) {
        final Map<Integer, Double> meanDeltas = history.stream()
                .filter(event -> event.getPrDelta() >= 0)
                .collect(Collectors.groupingBy(Event::userId, Collectors.averagingDouble(Event::getPrDelta)));
        final double totalMeanDelta = history.stream()
                .mapToDouble(Event::getPrDelta)
                .filter(x -> x >= 0)
                .average().orElse(-1);
        final TIntDoubleMap initialValues =
                new TIntDoubleHashMap(8, 0.5f, -1, 1 / totalMeanDelta);
        meanDeltas.forEach((userId, meanDelta) -> initialValues.put(userId, 1 / meanDelta));
        return initialValues;
    }
}
