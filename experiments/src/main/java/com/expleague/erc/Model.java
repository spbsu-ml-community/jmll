package com.expleague.erc;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.util.*;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

public class Model {
    private final int dim;
    private final double decayRate;
    private final double beta;
    private final double eps;
    private final double otherItemImportance;
    private final DoubleUnaryOperator lambdaTransform;
    private final DoubleUnaryOperator lambdaDerivativeTransform;
    private final LambdaStrategyFactory lambdaStrategyFactory;
    private TIntObjectMap<Vec> userEmbeddings;
    private TIntObjectMap<Vec> itemEmbeddings;
    private boolean dataInitialized;
    private int dataSize;
    private TIntSet userIds;
    private TIntSet itemIds;
    private int[] userIdsArray;
    private int[] itemIdsArray;
    private Vec zeroVec;

    public Model(final int dim, final double beta, final double eps, final double otherItemImportance,
                 final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                 final LambdaStrategyFactory lambdaStrategyFactory) {
        this(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory, null, null);
    }

    public Model(final int dim, final double beta, final double eps, final double otherItemImportance,
                 final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                 final LambdaStrategyFactory lambdaStrategyFactory, final TIntObjectMap<Vec> usersEmbeddingsPrior,
                 final TIntObjectMap<Vec> itemsEmbeddingsPrior) {
        this.dim = dim;
        decayRate = 0.97;
        this.beta = beta;
        this.eps = eps;
        this.otherItemImportance = otherItemImportance;
        this.lambdaTransform = lambdaTransform;
        this.lambdaDerivativeTransform = lambdaDerivativeTransform;
        this.lambdaStrategyFactory = lambdaStrategyFactory;
        this.userEmbeddings = usersEmbeddingsPrior;
        this.itemEmbeddings = itemsEmbeddingsPrior;
        dataInitialized = false;
        zeroVec = new ArrayVec(dim);
        VecTools.fill(zeroVec, 0);
    }

    private Vec makeEmbedding(final FastRandom randomGenerator, final double embMean) {
        Vec embedding = VecTools.copy(zeroVec);
        VecTools.fillGaussian(embedding, randomGenerator);
        VecTools.scale(embedding, embMean / 2);
        VecTools.adjust(embedding, embMean);
        for (int i = 0; i < dim; ++i) {
            embedding.set(i, Math.abs(embedding.get(i)));
        }
        return embedding;
    }

    private TIntObjectMap<Vec> initEmbeddings(final TIntSet ids, final double embMean) {
        final FastRandom randomGenerator = new FastRandom();
        final TIntObjectMap<Vec> embeddings = new TIntObjectHashMap<>();
        ids.forEach(itemId -> {
            embeddings.put(itemId, makeEmbedding(randomGenerator, embMean));
            return true;
        });
        return embeddings;
    }

    private void initializeEmbeddings(final List<Event> events) {
        if (dataInitialized) {
            return;
        }
        dataSize = events.size();
        userIds = new TIntHashSet();
        itemIds = new TIntHashSet();
        events.stream().map(Event::userId).forEach(userIds::add);
        events.stream().map(Event::itemId).forEach(itemIds::add);
        userIdsArray = userIds.toArray();
        itemIdsArray = itemIds.toArray();

        double itemDeltaMean = events.stream()
                .filter(event -> event.getPrDelta() >= 0)
                .collect(Collectors.averagingDouble(Event::getPrDelta));
        double embMean = Math.sqrt(1 / itemDeltaMean) / dim;
        System.out.println("Embedding mean = " + embMean);
        if (userEmbeddings == null) {
            userEmbeddings = initEmbeddings(userIds, embMean);
        }
        if (itemEmbeddings == null) {
            itemEmbeddings = initEmbeddings(itemIds, embMean);
        }
        dataInitialized = true;
    }

    private double logLikelihood(final List<Event> events) {
        double logLikelihood = 0.;
        final TIntObjectMap<TIntSet> seenItems = new TIntObjectHashMap<>();
        for (final int userId : userIds.toArray()) {
            seenItems.put(userId, new TIntHashSet());
        }
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        for (final Event event : events) {
            final int userId = event.userId();
            final int itemId = event.itemId();
            if (!seenItems.get(userId).contains(itemId)) {
                seenItems.get(userId).add(itemId);
                lambdasByItem.accept(event);
                continue;
            }
            // handle last events at another way
            if (!event.isFinish()) {
                final double lambda = lambdasByItem.getLambda(userId, itemId);
                final double transformedLambda = lambdaTransform.applyAsDouble(lambda);
                final double logLikelihoodDelta = Math.log(-Math.exp(-transformedLambda * (event.getPrDelta() + eps)) +
                        Math.exp(-transformedLambda * Math.max(0, event.getPrDelta() - eps)));
                if (!Double.isNaN(logLikelihoodDelta)) {
                    logLikelihood += logLikelihoodDelta;
                }
                lambdasByItem.accept(event);
            }
        }
        return logLikelihood;
    }

    private class Derivative {
        private final TIntObjectMap<Vec> userDerivatives;
        private final TIntObjectMap<Vec> itemDerivatives;

        private Derivative(final TIntObjectMap<Vec> userDerivatives, final TIntObjectMap<Vec> itemDerivatives) {
            this.userDerivatives = userDerivatives;
            this.itemDerivatives = itemDerivatives;
        }

        private TIntObjectMap<Vec> getUserDerivatives() {
            return userDerivatives;
        }

        private TIntObjectMap<Vec> getItemDerivatives() {
            return itemDerivatives;
        }
    }

    private Derivative logLikelihoodDerivative(final List<Event> events) {
        final TIntObjectMap<Vec> userDerivatives = new TIntObjectHashMap<>();
        final TIntObjectMap<TIntSet> seenItems = new TIntObjectHashMap<>();
        for (final int userId : userIds.toArray()) {
            userDerivatives.put(userId, VecTools.copy(zeroVec));
            seenItems.put(userId, new TIntHashSet());
        }
        final TIntObjectMap<Vec> itemDerivatives = new TIntObjectHashMap<>();
        for (final int itemId : itemIds.toArray()) {
            itemDerivatives.put(itemId, VecTools.copy(zeroVec));
        }
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        for (final Event event : events) {
            if (!seenItems.get(event.userId()).contains(event.itemId())) {
                seenItems.get(event.userId()).add(event.itemId());
                lambdasByItem.accept(event);
                continue;
            }
            if (!event.isFinish()) {
                updateInnerEventDerivative(lambdasByItem, event, userDerivatives, itemDerivatives);
                lambdasByItem.accept(event);
            }
        }
        return new Derivative(userDerivatives, itemDerivatives);
    }

    private void updateInnerEventDerivative(LambdaStrategy lambdasByItem, final Event event,
                                            TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives) {
        final double lambda = lambdasByItem.getLambda(event.userId(), event.itemId());
        final Vec lambdaDerivativeUser =
                lambdasByItem.getLambdaUserDerivative(event.userId(), event.itemId());
        final TIntObjectMap<Vec> lambdaDerivativeItems =
                lambdasByItem.getLambdaItemDerivative(event.userId(), event.itemId());
        final double transformedLambda = lambdaTransform.applyAsDouble(lambda);
        final double tau = event.getPrDelta();
        final double exp_plus = Math.exp(-transformedLambda * (tau + eps));
        final double exp_minus = Math.exp(-transformedLambda * Math.max(0, tau - eps));
        final double commonPart = lambdaDerivativeTransform.applyAsDouble(lambda) *
                ((tau + eps) * exp_plus - Math.max(0, tau - eps) * exp_minus) / (-exp_plus + exp_minus);
        if (!Double.isNaN(commonPart)) {
            VecTools.scale(lambdaDerivativeUser, commonPart);
            VecTools.append(userDerivatives.get(event.userId()), lambdaDerivativeUser);
            for (final int itemId : lambdaDerivativeItems.keys()) {
                final Vec derivative = lambdaDerivativeItems.get(itemId);
                VecTools.scale(derivative, commonPart);
                VecTools.append(itemDerivatives.get(itemId), derivative);
            }
        } else {
//            System.out.println("overflow");
        }
    }

    public void fit(final List<Event> events, final double learningRate, final int iterationsNumber,
                    final List<Event> evaluationEvents, final boolean verbose) {
        initializeEmbeddings(events);
        optimizeSGD(events, learningRate, iterationsNumber, evaluationEvents, verbose);
    }

    private void optimizeSGD(final List<Event> events, final double learningRate, final int iterationsNumber,
                             final List<Event> evaluationEvents, final boolean verbose) {
        double lr = learningRate / dataSize;
        Map<Integer, List<Event>> eventsByUser = new HashMap<>(userIds.size());
        for (final int userId : userIdsArray) {
            eventsByUser.put(userId, new ArrayList<>());
        }
        for (final Event event : events) {
            List<Event> events1 = eventsByUser.get(event.userId());
            events1.add(event);
        }

        MetricsCalculator metricsCalculator = new MetricsCalculator(events, evaluationEvents);
        for (int i = 0; i < iterationsNumber; ++i) {
            for (final List<Event> userEvents : eventsByUser.values()) {
                Derivative derivative = logLikelihoodDerivative(userEvents);
                for (final int userId : userIdsArray) {
                    Vec userDerivative = derivative.getUserDerivatives().get(userId);
                    VecTools.scale(userDerivative, lr);
                    VecTools.append(userEmbeddings.get(userId), userDerivative);
                }
                for (final int itemId : itemIdsArray) {
                    Vec itemDerivative = derivative.getItemDerivatives().get(itemId);
                    VecTools.scale(itemDerivative, lr);
                    VecTools.append(itemEmbeddings.get(itemId), itemDerivative);
                }
            }
            lr *= decayRate;
            if (verbose) {
                System.out.println(i + "-th iter, ll = " + logLikelihood(events));
                if (!evaluationEvents.isEmpty()) {
                    metricsCalculator.printMetrics(this);
                }
                System.out.println();
                System.out.println();
            }
        }
    }

    public class Applicable {
        private final LambdaStrategy lambdaStrategy;

        private Applicable() {
            lambdaStrategy = lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        }

        public void accept(final Event event) {
            lambdaStrategy.accept(event);
        }

        public double getLambda(final int userId, final int itemId) {
            return lambdaTransform.applyAsDouble(lambdaStrategy.getLambda(userId, itemId));
        }

        public double timeDelta(final int userId, final int itemId) {
            return 1 / getLambda(userId, itemId);
        }

        public Applicable fit(final List<Event> history) {
            for (Event event : history) {
                accept(event);
            }
            return this;
        }

        public TIntObjectMap<Vec> getUserEmbeddings() {
            return userEmbeddings;
        }

        public TIntObjectMap<Vec> getItemEmbeddings() {
            return itemEmbeddings;
        }
    }

    public Applicable getApplicable(final List<Event> events) {
        Applicable applicable = new Applicable();
        if (events != null) {
            applicable.fit(events);
        }
        return applicable;
    }

    public Applicable getApplicable() {
        return new Applicable();
    }
}
