package com.expleague.erc;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;

import java.util.*;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Model {
    private final int dimensionality;
    private final double decayRate;
    private final double beta;
    private final double eps;
    private final double otherProjectImportance;
    private final DoubleUnaryOperator lambdaTransform;
    private final DoubleUnaryOperator lambdaDerivativeTransform;
    private final LambdaStrategyFactory lambdaStrategyFactory;
    private Map<String, ArrayVec> userEmbeddings;
    private Map<String, ArrayVec> itemEmbeddings;
    private boolean dataInitialized;
    private int dataSize;
    private Set<String> userIds;
    private Set<String> itemIds;

    public Model(int dimensionality, double beta, double eps, double otherItemImportance,
                 DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                 LambdaStrategyFactory lambdaStrategyFactory, Map<String, ArrayVec> usersEmbeddingsPrior,
                 Map<String, ArrayVec> projectsEmbeddingsPrior) {
        this.dimensionality = dimensionality;
        decayRate = 1;
        this.beta = beta;
        this.eps = eps;
        this.otherProjectImportance = otherItemImportance;
        this.lambdaTransform = lambdaTransform;
        this.lambdaDerivativeTransform = lambdaDerivativeTransform;
        this.lambdaStrategyFactory = lambdaStrategyFactory;
        this.userEmbeddings = usersEmbeddingsPrior;
        this.itemEmbeddings = projectsEmbeddingsPrior;
        dataInitialized = false;
    }

    private ArrayVec makeEmbedding(FastRandom randomGenerator, double embMean) {
        ArrayVec embedding = new ArrayVec(dimensionality);
        VecTools.fillGaussian(embedding, randomGenerator);
        embedding.scale(embMean / 2);
        VecTools.adjust(embedding, embMean);
        for (int i = 0; i < dimensionality; ++i) {
            embedding.set(i, Math.abs(embedding.get(i)));
        }
        return embedding;
    }

    private void initializeData(List<Event> history) {
        if (dataInitialized) {
            return;
        }
        dataSize = history.size();
        userIds = history.stream().map(Event::getUid).collect(Collectors.toSet());
        itemIds = history.stream().map(Event::getPid).collect(Collectors.toSet());

        double itemDeltaMean = history.stream()
                .filter(event -> event.getPrDelta() != null)
                .collect(Collectors.averagingDouble(Event::getPrDelta));
        double embMean = Math.sqrt(1 / itemDeltaMean) / dimensionality;
        System.out.println("Embedding mean =" + embMean);
        if (userEmbeddings == null) {
            FastRandom randomGenerator = new FastRandom();
            userEmbeddings = new HashMap<>();
            for (String user: userIds) {
                userEmbeddings.put(user, makeEmbedding(randomGenerator, embMean));
            }
        }
        if (itemEmbeddings == null) {
            FastRandom randomGenerator = new FastRandom();
            itemEmbeddings = new HashMap<>();
            for (String item: itemIds) {
                itemEmbeddings.put(item, makeEmbedding(randomGenerator, embMean));
            }
        }
        dataInitialized = true;
    }

    private double logLikelihood(List<Event> history) {
        initializeData(history);
        double logLikelihood = 0.;
        Map<String, Set<String>> done_projects = userIds.stream()
                .collect(Collectors.toMap(Function.identity(), (userId) -> new HashSet<>()));
        LambdaStrategy lambdasByProject =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherProjectImportance);
        List<Event> lastTimeEvents = new ArrayList<>();
        for (Event event: history) {
            if (!done_projects.get(event.getUid()).contains(event.getPid())) {
                done_projects.get(event.getUid()).add(event.getPid());
                lambdasByProject.accept(event);
                continue;
            }
            if (event.getNTasks() != 0) {
                double lambda = lambdasByProject.getLambda(event.getUid(), event.getPid());
                double transformedLambda = lambdaTransform.applyAsDouble(lambda);
                double logLikelihoodDelta = Math.log(-Math.exp(-transformedLambda * (event.getPrDelta() + eps)) +
                        Math.exp(-transformedLambda * Math.max(0, event.getPrDelta() - eps)));
//                TODO: check for overflow
                logLikelihood += logLikelihoodDelta;
                lambdasByProject.accept(event);
            } else {
                lastTimeEvents.add(event);
            }
        }
        for (Event event: lastTimeEvents) {
            double lambda = lambdasByProject.getLambda(event.getUid(), event.getPid());
            double transformedLambda = lambdaTransform.applyAsDouble(lambda);
            logLikelihood += -transformedLambda * event.getPrDelta();
        }
        return logLikelihood;
    }

    private class Derivative {
        private final Map<String, ArrayVec> userDerivatives;
        private final Map<String, ArrayVec> itemDerivatives;

        private Derivative(Map<String, ArrayVec> userDerivatives, Map<String, ArrayVec> itemDerivatives) {
            this.userDerivatives = userDerivatives;
            this.itemDerivatives = itemDerivatives;
        }

        private Map<String, ArrayVec> getUserDerivatives() {
            return userDerivatives;
        }

        private Map<String, ArrayVec> getItemDerivatives() {
            return itemDerivatives;
        }
    }

    private Derivative logLikelihoodDerivative(List<Event> history) {
        initializeData(history);
        Map<String, ArrayVec> userDerivatives = userIds.stream()
                .collect(Collectors.toMap(Function.identity(), (userId) -> {
                    ArrayVec derivative = new ArrayVec(dimensionality);
                    derivative.fill(0.);
                    return derivative;
                }));
        Map <String, ArrayVec> itemDerivatives = itemIds.stream()
                .collect(Collectors.toMap(Function.identity(), (userId) -> {
                    ArrayVec derivative = new ArrayVec(dimensionality);
                    derivative.fill(0.);
                    return derivative;
                }));
        Map<String, Set<String>> done_projects = userIds.stream()
                .collect(Collectors.toMap(Function.identity(), (userId) -> new HashSet<>()));
        LambdaStrategy lambdasByProject =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherProjectImportance);
        List<Event> lastTimeEvents = new ArrayList<>();
        for (Event event: history) {
            if (!done_projects.get(event.getUid()).contains(event.getPid())) {
                done_projects.get(event.getUid()).add(event.getPid());
                lambdasByProject.accept(event);
                continue;
            }
            if (event.getNTasks() != 0) {
                double lambda = lambdasByProject.getLambda(event.getUid(), event.getPid());
                ArrayVec lambdaDerivativeUser =
                        lambdasByProject.getLambdaUserDerivative(event.getUid(), event.getPid());
                Map <String, ArrayVec> lambdaDerivativeProjects =
                        lambdasByProject.getLambdaProjectDerivative(event.getUid(), event.getPid());
                double transformedLambda = lambdaTransform.applyAsDouble(lambda);
                double tau = event.getPrDelta();
                double exp_plus = Math.exp(-transformedLambda * (tau + eps));
                double exp_minus = Math.exp(-transformedLambda * Math.max(0, tau - eps));
                double commonPart = lambdaDerivativeTransform.applyAsDouble(lambda) *
                        ((tau + eps) * exp_plus - Math.max(0, tau - eps) * exp_minus) / (-exp_plus + exp_minus);
//                TODO: check for overflow
                lambdaDerivativeUser.scale(commonPart);
                userDerivatives.get(event.getUid()).add(lambdaDerivativeUser);
                lambdaDerivativeProjects.forEach((itemId, derivative) -> {
                    derivative.scale(commonPart);
                    itemDerivatives.get(itemId).add(derivative);
                });
                lambdasByProject.accept(event);
            } else {
                lastTimeEvents.add(event);
            }
        }
        for (Event event: lastTimeEvents) {
            double lambda = lambdasByProject.getLambda(event.getUid(), event.getPid());
            ArrayVec lambdaDerivativeUser =
                    lambdasByProject.getLambdaUserDerivative(event.getUid(), event.getPid());
            Map <String, ArrayVec> lambdaDerivativeProjects =
                    lambdasByProject.getLambdaProjectDerivative(event.getUid(), event.getPid());
            double commonPart = lambdaDerivativeTransform.applyAsDouble(lambda) * event.getPrDelta();
            lambdaDerivativeUser.scale(-commonPart);
            userDerivatives.get(event.getUid()).add(lambdaDerivativeUser);
            lambdaDerivativeProjects.forEach((itemId, derivative) -> {
                derivative.scale(-commonPart);
                itemDerivatives.get(itemId).add(derivative);
            });
        }
        return new Derivative(userDerivatives, itemDerivatives);
    }

    void optimizeSGD(List<Event> data, double learningRate, int iterationsNumber, List<Event> evaluationData,
                     boolean verbose) {
        initializeData(data);
        learningRate /= dataSize;
        for (int i = 0; i < iterationsNumber; ++i) {
            Derivative derivative = logLikelihoodDerivative(data);
            for (String userId: userIds) {
                ArrayVec userDerivative = derivative.getUserDerivatives().get(userId);
                userDerivative.scale(learningRate);
                userEmbeddings.get(userId).add(userDerivative);
            }
            for (String itemId: itemIds) {
                ArrayVec itemDerivative = derivative.getItemDerivatives().get(itemId);
                itemDerivative.scale(learningRate);
                itemEmbeddings.get(itemId).add(itemDerivative);
            }
            learningRate *= decayRate;
            if (verbose) {
                System.out.println(i + "{}-th iter, ll = {}" + logLikelihood(data));
                if (evaluationData != null) {
                    // TODO: print metrics
                }
                System.out.println();
            }
        }
    }
}
