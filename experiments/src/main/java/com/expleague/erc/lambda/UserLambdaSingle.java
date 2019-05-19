package com.expleague.erc.lambda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ModelCombined;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.util.List;

public class UserLambdaSingle implements UserLambda {
    private static final double EPS = 0.2;

    private final Vec userEmbedding;
    private final TIntObjectMap<Vec> itemEmbeddings;
    private final double beta;
    private final int dim;
    private final double initialLambda;

    private double lambda;
    private double currentTime;
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
        lastTimeOfItems = new TIntDoubleHashMap();

        initialLambda = initialValue;
//        lambda = initialLambda;
        lambda = 0.;
        userDerivative = new ArrayVec(dim);
        itemDerivatives = new TIntObjectHashMap<>();
    }

    @Override
    public void reset() {
        lambda = 0.;
//        lambda = initialLambda;
        currentTime = 0.;
        lastTimeOfItems.clear();
        itemDerivatives.clear();
        VecTools.fill(userDerivative, 0);
    }

    @Override
    public final void update(final int itemId, double timeDelta) {
        timeDelta = 1.;
        final Vec itemEmbedding = itemEmbeddings.get(itemId);
        if (!lastTimeOfItems.containsKey(itemId)) {
            lastTimeOfItems.put(itemId, currentTime);
            itemDerivatives.put(itemId, new ArrayVec(dim));
        }

        // Updating lambda
        final double e = Math.exp(-beta * timeDelta);
        final double interactionEffect = VecTools.multiply(userEmbedding, itemEmbedding);
        lambda = e * lambda + interactionEffect;

        // Updating user derivative
        VecTools.scale(userDerivative, e);
        VecTools.append(userDerivative, itemEmbedding);

        // Updating item derivative
        final Vec itemDerivative = itemDerivatives.get(itemId);
        final double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(itemId)));
        VecTools.scale(itemDerivative, decay);
        VecTools.append(itemDerivative, userEmbedding);

        lastTimeOfItems.put(itemId, currentTime);
        currentTime += timeDelta;
    }

    @Override
    public final double getLambda(final int itemId) {
        return initialLambda + lambda;
    }

    public final double getLambda() {
        return initialLambda + lambda;
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
            final Vec curDerivative = VecTools.copy(itemDerivative);
            final double decay = Math.exp(-beta * (currentTime - lastTimeOfItems.get(curItemId)));
            VecTools.scale(curDerivative, decay);
            derivative.put(curItemId, curDerivative);
            return true;
        });
        return derivative;
    }

    @Override
    public TIntObjectMap<Vec> getLambdaItemsDerivative(int itemId) {
        throw new UnsupportedOperationException();
    }

    public static TIntDoubleMap makeUserLambdaInitialValues(final List<Event> history) {
        final TIntIntMap userBorders = ModelCombined.findMinHourInDay(history);
        final TIntObjectMap<TDoubleList> userDeltas = new TIntObjectHashMap<>();
        for (final Session session : DataPreprocessor.groupEventsToSessions(history)) {
            final int userId = session.userId();
            if (!userDeltas.containsKey(userId)) {
                userDeltas.put(userId, new TDoubleArrayList());
            }
            if (Util.forPrediction(session)) {
                final int daysFromPrevSession = Util.getDaysFromPrevSession(session, userBorders.get(session.userId()));
                userDeltas.get(userId).add(daysFromPrevSession);
            }
        }
        final double[] intervals = DataPreprocessor.groupEventsToSessions(history).stream()
                .filter(Util::forPrediction)
                .mapToDouble(session -> Util.getDaysFromPrevSession(session, userBorders.get(session.userId())))
                .sorted().toArray();
        final double commonConst = Math.max(intervals[intervals.length / 2], EPS);
        TIntDoubleMap constants = new TIntDoubleHashMap();
        userDeltas.forEachEntry((userId, deltas) -> {
            deltas.sort();
            double[] dels = deltas.toArray();
            double del;
            if (dels.length == 0) {
                del = commonConst;
            } else {
                del = dels[dels.length / 2];
                if (del == 0) {
                    del = EPS;
                }
            }
            constants.put(userId, 1 / del);
            return true;
        });
        return constants;
    }

//    public static TIntDoubleMap makeUserLambdaInitialValues(final List<Event> history, final double lambdaMultiplier) {
//        final Map<Integer, Double> meanDeltas = DataPreprocessor.groupEventsToSessions(history).stream()
//                .filter(session -> session.getDelta() >= 0 && session.getDelta() < Util.CHURN_THRESHOLD)
//                .collect(Collectors.groupingBy(Session::userId, Collectors.averagingDouble(session ->
//                        session.getDelta() * lambdaMultiplier)));
//        final double totalMeanDelta = DataPreprocessor.groupEventsToSessions(history).stream()
//                .mapToDouble(Session::getDelta)
//                .filter(x -> x >= 0 && x < Util.CHURN_THRESHOLD)
//                .map(delta -> delta * lambdaMultiplier)
//                .average().orElse(-1);
//        final TIntDoubleMap initialValues =
//                new TIntDoubleHashMap(8, 0.5f, -1, 1 / totalMeanDelta);
//        meanDeltas.forEach((userId, meanDelta) -> initialValues.put(userId, 1 / meanDelta));
//        return initialValues;
//    }
}
