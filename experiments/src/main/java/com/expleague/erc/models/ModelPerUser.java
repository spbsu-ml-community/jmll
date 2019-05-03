package com.expleague.erc.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TLongDoubleHashMap;
import gnu.trove.set.TLongSet;
import gnu.trove.set.hash.TLongHashSet;

import java.util.List;
import java.util.function.DoubleUnaryOperator;

public class ModelPerUser extends Model {
    public ModelPerUser(int dim, double beta, double eps, double otherItemImportance,
                        DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                        LambdaStrategyFactory lambdaStrategyFactory, TIntObjectMap<Vec> usersEmbeddingsPrior,
                        TIntObjectMap<Vec> itemsEmbeddingsPrior) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                usersEmbeddingsPrior, itemsEmbeddingsPrior);
    }

    @Override
    public void logLikelihoodDerivative(final List<Event> events,
                                        final TIntObjectMap<Vec> userDerivatives,
                                        final TIntObjectMap<Vec> itemDerivatives) {
        final double observationEnd = events.get(events.size() - 1).getTs();
        final TLongSet seenPairs = new TLongHashSet();
        for (final int userId : userIds.toArray()) {
            userDerivatives.put(userId, new ArrayVec(dim));
        }
        for (final int itemId : itemIds.toArray()) {
            itemDerivatives.put(itemId, new ArrayVec(dim));
        }
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
        final TIntDoubleMap userLastVisitTimes = new TIntDoubleHashMap();
        for (final Session session : DataPreprocessor.groupToSessions(events)) {
            final long pairId = session.getPair();
            final int userId = session.userId();
            final double time = session.getTs();
            if (!seenPairs.contains(pairId)) {
                seenPairs.add(pairId);
                lambdasByItem.accept(session);
            } else {
                updateDerivativeInnerEvent(lambdasByItem, userId, session.itemId(),
                        time - userLastVisitTimes.get(userId),
                        userDerivatives, itemDerivatives);
                lambdasByItem.accept(session);
                lastVisitTimes.put(pairId, time);
            }
            userLastVisitTimes.put(userId, time);
        }
        for (long pairId: lastVisitTimes.keys()) {
            final int userId = Util.extractUserId(pairId);
            final int itemId = Util.extractItemId(pairId);
            updateDerivativeLastEvent(lambdasByItem, userId, itemId, observationEnd - lastVisitTimes.get(pairId),
                    userDerivatives, itemDerivatives);
        }
    }
}
