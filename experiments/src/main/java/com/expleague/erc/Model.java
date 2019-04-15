package com.expleague.erc;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TLongDoubleHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

public class Model {
    private static final int SAVE_PERIOD = 20;

    private final int dim;
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

    public void initializeEmbeddings(final List<Event> events) {
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
//                .collect(Collectors.averagingDouble(event -> Math.log(event.getPrDelta())));
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
        final double observationEnd = events.get(events.size() - 1).getTs();
        double logLikelihood = 0.;
        final TIntObjectMap<TIntSet> seenItems = new TIntObjectHashMap<>();
        for (final int userId : userIds.toArray()) {
            seenItems.put(userId, new TIntHashSet());
        }
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
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
                double prDelta = event.getPrDelta();
//                prDelta = Math.log(prDelta);
                final double logLikelihoodDelta =
                        Math.log(-Math.exp(-transformedLambda * (prDelta + eps)) +
                        Math.exp(-transformedLambda * Math.max(0, prDelta - eps)));
                if (Double.isFinite(logLikelihoodDelta)) {
                    logLikelihood += logLikelihoodDelta;
                }
                lambdasByItem.accept(event);
                lastVisitTimes.put(Util.combineIds(userId, itemId), event.getTs());
            }
        }
        for (long pairId : lastVisitTimes.keys()) {
            final int userId = (int)(pairId >> 32);
            final int itemId = (int)pairId;
            final double lambda = lambdasByItem.getLambda(userId, itemId);
            final double transformedLambda = lambdaTransform.applyAsDouble(lambda);
            final double logLikelihoodDelta = -transformedLambda * observationEnd - lastVisitTimes.get(pairId);
//            final double logLikelihoodDelta = -transformedLambda * Math.log(observationEnd - lastVisitTimes.get(pairId));
            if (Double.isFinite(logLikelihoodDelta)) {
                logLikelihood += logLikelihoodDelta;
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
        final double observationEnd = events.get(events.size() - 1).getTs();
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
        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
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
            lastVisitTimes.put(Util.combineIds(event.userId(), event.itemId()), event.getTs());
        }
        for (long pairId: lastVisitTimes.keys()) {
            final int userId = Util.extractUserId(pairId);
            final int itemId = Util.extractItemId(pairId);
            updateDerivativeLastEvent(lambdasByItem, userId, itemId, observationEnd - lastVisitTimes.get(pairId),
                    userDerivatives, itemDerivatives);
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
        double tau = event.getPrDelta();
//        tau = Math.log(tau);
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

    private void updateDerivativeLastEvent(LambdaStrategy lambdasByItem, int userId, int itemId, double tau,
                                           TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives) {
        final Vec lambdaDerivativeUser =
                lambdasByItem.getLambdaUserDerivative(userId, itemId);
        final TIntObjectMap<Vec> lambdaDerivativeItems =
                lambdasByItem.getLambdaItemDerivative(userId, itemId);

        VecTools.scale(lambdaDerivativeUser, -tau);
        VecTools.append(userDerivatives.get(userId), lambdaDerivativeUser);

        Vec lambdaDerivativeItem = lambdaDerivativeItems.get(itemId);
        VecTools.scale(lambdaDerivativeItem, -tau);
        VecTools.append(itemDerivatives.get(itemId), lambdaDerivativeItem);
    }

    public void fit(final List<Event> events, final double learningRate, final int iterationsNumber,
                    final List<Event> evaluationEvents, final double decay, final boolean verbose,
                    MetricsCalculator metricsCalculator, String savePathPrev) {
        initializeEmbeddings(events);
        optimizeGD(events, learningRate, iterationsNumber, evaluationEvents, decay, verbose, metricsCalculator, savePathPrev);
    }

    private void optimizeSGD(final List<Event> events, final double learningRate, final int iterationsNumber,
                             final List<Event> evaluationEvents, final double decay, final boolean verbose,
                             MetricsCalculator metricsCalculator) {
        double lr = learningRate / dataSize;
        Map<Integer, List<Event>> eventsByUser = new HashMap<>(userIds.size());
        for (final int userId : userIdsArray) {
            eventsByUser.put(userId, new ArrayList<>());
        }
        for (final Event event : events) {
            List<Event> events1 = eventsByUser.get(event.userId());
            events1.add(event);
        }

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
            lr *= decay;
            printWhileOptimization(events, evaluationEvents, i, verbose, metricsCalculator);
        }
    }

    private void optimizeGD(final List<Event> events, final double learningRate, final int iterationsNumber,
                            final List<Event> evaluationEvents, final double decay, final boolean verbose,
                            MetricsCalculator metricsCalculator, String savePathPref) {
        double lr = learningRate / dataSize;
        for (int i = 0; i < iterationsNumber; ++i) {
            Derivative derivative = logLikelihoodDerivative(events);
            for (final int userId : userIds.toArray()) {
                Vec userDerivative = derivative.getUserDerivatives().get(userId);
                VecTools.scale(userDerivative, lr);
                VecTools.append(userEmbeddings.get(userId), userDerivative);
            }
            for (final int itemId : itemIds.toArray()) {
                Vec itemDerivative = derivative.getItemDerivatives().get(itemId);
                VecTools.scale(itemDerivative, lr);
                VecTools.append(itemEmbeddings.get(itemId), itemDerivative);
            }
            lr *= decay;
            printWhileOptimization(events, evaluationEvents, i, verbose, metricsCalculator);
            checkpoint(i + 1, savePathPref);
        }
    }

    private void printWhileOptimization(final List<Event> events, final List<Event> evaluationEvents,
                                        final int iteration, final boolean verbose,
                                        final MetricsCalculator metricsCalculator) {
        if (verbose) {
            System.out.println(iteration + "-th iter, ll = " + logLikelihood(events));
            if (evaluationEvents != null && !evaluationEvents.isEmpty()) {
                try {
                    metricsCalculator.writeLambdas(this);
                    final MetricsCalculator.Summary summary = metricsCalculator.calculateSummary(this);
                    System.out.println(summary);
                    summary.writeSpus();
                } catch (ExecutionException | InterruptedException | IOException e) {
                    e.printStackTrace();
                }
            }
            System.out.println();
        }
    }

    private void checkpoint(int iterationsPassed, String savePathPref) {
        if (savePathPref != null && iterationsPassed % SAVE_PERIOD == 0) {
            try {
                write(Files.newOutputStream(Paths.get(savePathPref + "_" + iterationsPassed)));
            } catch (IOException e) {
                e.printStackTrace();
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

//        public double timeDeltaLog(final int userId, final int itemId) {
//            return 1 / getLambda(userId, itemId);
//        }

        public double timeDelta(final int userId, final int itemId) {
//            return Math.exp(1 / getLambda(userId, itemId));
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

    private static Map<Integer, double[]> embeddingsToSerializable(final TIntObjectMap<Vec> embeddings) {
        return Arrays.stream(embeddings.keys())
                .boxed()
                .collect(Collectors.toMap(x -> x, x -> embeddings.get(x).toArray()));
    }

    private static TIntObjectMap<Vec> embeddingsFromSerializable(final Map<Integer, double[]> serMap) {
        final TIntObjectMap<Vec> embeddings = new TIntObjectHashMap<>();
        serMap.forEach((id, embedding) -> embeddings.put(id, new ArrayVec(embedding)));
        return embeddings;
    }

    public void write(final OutputStream stream) throws IOException {
        final ObjectOutputStream objectOutputStream = new ObjectOutputStream(stream);
        objectOutputStream.writeInt(dim);
        objectOutputStream.writeDouble(beta);
        objectOutputStream.writeDouble(eps);
        objectOutputStream.writeDouble(otherItemImportance);
        objectOutputStream.writeObject(lambdaTransform);
        objectOutputStream.writeObject(lambdaDerivativeTransform);
        objectOutputStream.writeObject(lambdaStrategyFactory);
        objectOutputStream.writeObject(embeddingsToSerializable(userEmbeddings));
        objectOutputStream.writeObject(embeddingsToSerializable(itemEmbeddings));
    }

    public static Model load(final InputStream stream) throws IOException, ClassNotFoundException {
        final ObjectInputStream objectInputStream = new ObjectInputStream(stream);
        final int dim = objectInputStream.readInt();
        final double beta = objectInputStream.readDouble();
        final double eps = objectInputStream.readDouble();
        final double otherItemImportance = objectInputStream.readDouble();
        final DoubleUnaryOperator lambdaTransform = (DoubleUnaryOperator) objectInputStream.readObject();
        final DoubleUnaryOperator lambdaDerivativeTransform = (DoubleUnaryOperator) objectInputStream.readObject();
        final LambdaStrategyFactory lambdaStrategyFactory = (LambdaStrategyFactory) objectInputStream.readObject();
        final TIntObjectMap<Vec> userEmbeddings =
                embeddingsFromSerializable((Map<Integer, double[]>) objectInputStream.readObject());
        final TIntObjectMap<Vec> itemEmbeddings =
                embeddingsFromSerializable((Map<Integer, double[]>) objectInputStream.readObject());
        return new Model(dim, beta, eps, otherItemImportance, lambdaTransform,
                lambdaDerivativeTransform, lambdaStrategyFactory, userEmbeddings, itemEmbeddings);
    }
}
