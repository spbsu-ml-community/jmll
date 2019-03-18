package com.expleague.erc;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;

import java.util.*;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
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
    private Map<String, Vec> userEmbeddings;
    private Map<String, Vec> itemEmbeddings;
    private boolean dataInitialized;
    private int dataSize;
    private Set<String> userIds;
    private Set<String> itemIds;
    private Vec zeroVec;

    public Model(final int dim, final double beta, final double eps, final double otherItemImportance,
                 final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                 final LambdaStrategyFactory lambdaStrategyFactory, final Map<String, Vec> usersEmbeddingsPrior,
                 final Map<String, Vec> itemsEmbeddingsPrior) {
        this.dim = dim;
        decayRate = 1;
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

    private void initializeEmbeddings(final List<Event> events) {
        if (dataInitialized) {
            return;
        }
        dataSize = events.size();
        userIds = events.stream().map(Event::userId).collect(Collectors.toSet());
        itemIds = events.stream().map(Event::itemId).collect(Collectors.toSet());

        double itemDeltaMean = events.stream()
                .filter(event -> event.getPrDelta() != null)
                .collect(Collectors.averagingDouble(Event::getPrDelta));
        double embMean = Math.sqrt(1 / itemDeltaMean) / dim;
        System.out.println("Embedding mean =" + embMean);
        FastRandom randomGenerator = new FastRandom();
        if (userEmbeddings == null) {
            itemEmbeddings = userIds.stream().collect(Collectors.toMap(id -> id,
                    id -> makeEmbedding(randomGenerator, embMean)));
        }
        if (itemEmbeddings == null) {
            itemEmbeddings = itemIds.stream().collect(Collectors.toMap(id -> id,
                    id -> makeEmbedding(randomGenerator, embMean)));
        }
        dataInitialized = true;
    }

    private double logLikelihood(final List<Event> events) {
        initializeEmbeddings(events);
        double logLikelihood = 0.;
        final Map<String, Set<String>> seenItems = userIds.stream()
                .collect(Collectors.toMap(Function.identity(), (userId) -> new HashSet<>()));
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
//        final List<Event> lastTimeEvents = new ArrayList<>();
        for (final Event event : events) {
            final String userId = event.userId();
            final String itemId = event.itemId();
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
//                TODO: check for overflow
                logLikelihood += logLikelihoodDelta;
                lambdasByItem.accept(event);
            }
//            } else {
//                lastTimeEvents.add(event);
//            }
        }
//        for (final Event event : lastTimeEvents) {
//            final double lambda = lambdasByItem.getLambda(event.userId(), event.itemId());
//            final double transformedLambda = lambdaTransform.applyAsDouble(lambda);
//            logLikelihood += -transformedLambda * event.getPrDelta();
//        }
        return logLikelihood;
    }

    private class Derivative {
        private final Map<String, Vec> userDerivatives;
        private final Map<String, Vec> itemDerivatives;

        private Derivative(final Map<String, Vec> userDerivatives, final Map<String, Vec> itemDerivatives) {
            this.userDerivatives = userDerivatives;
            this.itemDerivatives = itemDerivatives;
        }

        private Map<String, Vec> getUserDerivatives() {
            return userDerivatives;
        }

        private Map<String, Vec> getItemDerivatives() {
            return itemDerivatives;
        }
    }

    private Derivative logLikelihoodDerivative(final List<Event> events) {
        initializeEmbeddings(events);
        final Map<String, Vec> userDerivatives = userIds.stream()
                .collect(Collectors.toMap(Function.identity(), userId -> VecTools.copy(zeroVec)));
        final Map<String, Vec> itemDerivatives = itemIds.stream()
                .collect(Collectors.toMap(Function.identity(), userId -> VecTools.copy(zeroVec)));
        final Map<String, Set<String>> seenItems = userIds.stream()
                .collect(Collectors.toMap(Function.identity(), userId -> new HashSet<>()));
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
//        final List<Event> lastTimeEvents = new ArrayList<>();
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
//            } else {
//                lastTimeEvents.add(event);
//            }
        }
//        for (final Event event : lastTimeEvents) {
//            final double lambda = lambdasByItem.getLambda(event.userId(), event.itemId());
//            final Vec lambdaDerivativeUser =
//                    lambdasByItem.getLambdaUserDerivative(event.userId(), event.itemId());
//            final Map<String, Vec> lambdaDerivativeItems =
//                    lambdasByItem.getLambdaItemDerivative(event.userId(), event.itemId());
//            final double commonPart = lambdaDerivativeTransform.applyAsDouble(lambda) * event.getPrDelta();
//            VecTools.scale(lambdaDerivativeUser, -commonPart);
//            VecTools.append(userDerivatives.get(event.userId()), lambdaDerivativeUser);
//            lambdaDerivativeItems.forEach((itemId, derivative) -> {
//                VecTools.scale(derivative, -commonPart);
//                VecTools.append(itemDerivatives.get(itemId), derivative);
//            });
//        }
        return new Derivative(userDerivatives, itemDerivatives);
    }

    private void updateInnerEventDerivative(LambdaStrategy lambdasByItem, final Event event,
                                            Map<String, Vec> userDerivatives, Map<String, Vec> itemDerivatives) {
        final double lambda = lambdasByItem.getLambda(event.userId(), event.itemId());
        final Vec lambdaDerivativeUser =
                lambdasByItem.getLambdaUserDerivative(event.userId(), event.itemId());
        final Map<String, Vec> lambdaDerivativeItems =
                lambdasByItem.getLambdaItemDerivative(event.userId(), event.itemId());
        final double transformedLambda = lambdaTransform.applyAsDouble(lambda);
        final double tau = event.getPrDelta();
        final double exp_plus = Math.exp(-transformedLambda * (tau + eps));
        final double exp_minus = Math.exp(-transformedLambda * Math.max(0, tau - eps));
        final double commonPart = lambdaDerivativeTransform.applyAsDouble(lambda) *
                ((tau + eps) * exp_plus - Math.max(0, tau - eps) * exp_minus) / (-exp_plus + exp_minus);
//                TODO: check for overflow
        VecTools.scale(lambdaDerivativeUser, commonPart);
        VecTools.append(userDerivatives.get(event.userId()), lambdaDerivativeUser);
        lambdaDerivativeItems.forEach((itemId, derivative) -> {
            VecTools.scale(derivative, commonPart);
            VecTools.append(itemDerivatives.get(itemId), derivative);
        });
    }

    public void fit(final List<Event> events, final double learningRate, final int iterationsNumber,
                    final List<Event> evaluationEvents, final boolean verbose) {
        optimizeSGD(events, learningRate, iterationsNumber, evaluationEvents, verbose);
    }

    private void optimizeSGD(final List<Event> events, final double learningRate, final int iterationsNumber,
                     final List<Event> evaluationEvents, final boolean verbose) {
        initializeEmbeddings(events);
        double lr = learningRate / dataSize;
        for (int i = 0; i < iterationsNumber; ++i) {
            Derivative derivative = logLikelihoodDerivative(events);
            for (String userId : userIds) {
                Vec userDerivative = derivative.getUserDerivatives().get(userId);
                VecTools.scale(userDerivative, lr);
                VecTools.append(userEmbeddings.get(userId), userDerivative);
            }
            for (String itemId : itemIds) {
                Vec itemDerivative = derivative.getItemDerivatives().get(itemId);
                VecTools.scale(itemDerivative, lr);
                VecTools.append(itemEmbeddings.get(itemId), itemDerivative);
            }
            lr *= decayRate;
            if (verbose) {
                System.out.println(i + "{}-th iter, ll = {}" + logLikelihood(events));
                if (evaluationEvents != null) {
                    Metrics.printMetrics(this, events, evaluationEvents);
                }
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

        public double getLambda(final String userId, final String itemId) {
            return lambdaTransform.applyAsDouble(lambdaStrategy.getLambda(userId, itemId));
        }

        public double timeDelta(final String userId, final String itemId) {
            return 1 / getLambda(userId, itemId);
        }

        public Applicable fit(final List<Event> history) {
            for (Event event : history) {
                accept(event);
            }
            return this;
        }

        public Map<String, Vec> getUserEmbeddings() {
            return userEmbeddings;
        }

        public Map<String, Vec> getItemEmbeddings() {
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
