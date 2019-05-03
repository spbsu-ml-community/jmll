package com.expleague.erc.models;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.erc.Event;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TLongDoubleHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.TLongSet;
import gnu.trove.set.hash.TIntHashSet;
import gnu.trove.set.hash.TLongHashSet;

import java.io.*;
import java.util.*;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

import static java.lang.Math.*;

public class Model {
//    private static final int SAVE_PERIOD = 20;
//    private static final double DEFAULT_BIAS = 1e-2;
//    private final double SCALE_SHARE = 0.3;

    protected final int dim;
    protected final double beta;
    protected final double eps;
    protected final double otherItemImportance;
    protected final DoubleUnaryOperator lambdaTransform;
    protected final DoubleUnaryOperator lambdaDerivativeTransform;
    protected final LambdaStrategyFactory lambdaStrategyFactory;
    protected TIntObjectMap<Vec> userEmbeddings;
    //    private TIntDoubleMap userBiases;
    protected TIntObjectMap<Vec> itemEmbeddings;
    //    private TIntDoubleMap itemBiases;
    protected TIntSet userIds;
    protected TIntSet itemIds;
    private int[] userIdsArray;
    private int[] itemIdsArray;

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
        userEmbeddings = usersEmbeddingsPrior;
//        this.userBiases = userBiases;
        itemEmbeddings = itemsEmbeddingsPrior;
//        this.itemBiases = itemBiases;
        userIds = userEmbeddings.keySet();
        userIdsArray = userIds.toArray();
        itemIds = itemEmbeddings.keySet();
        itemIdsArray = itemIds.toArray();
    }

    protected static Vec makeEmbedding(final FastRandom randomGenerator, final double embMean, final int dim) {
        Vec embedding = new ArrayVec(dim);

        VecTools.fillGaussian(embedding, randomGenerator);
        VecTools.scale(embedding, embMean / 2);
        VecTools.adjust(embedding, embMean);
        for (int i = 0; i < dim; ++i) {
            embedding.set(i, abs(embedding.get(i)));
        }

//        VecTools.fillUniform(embedding, randomGenerator, embMean * SCALE_SHARE);
//        VecTools.adjust(embedding, embMean);

        return embedding;
    }

    /*
        private TIntObjectMap<Vec> initEmbeddings(final TIntSet ids, final double embMean) {
            final FastRandom randomGenerator = new FastRandom();
            final TIntObjectMap<Vec> embeddings = new TIntObjectHashMap<>();
            ids.forEach(itemId -> {
                embeddings.put(itemId, makeEmbedding(randomGenerator, embMean, dim));
                return true;
            });
            return embeddings;
        }

        public void initializeEmbeddings(final List<Event> events) {
            if (dataInitialized) {
                return;
            }
            userIds = new TIntHashSet();
            itemIds = new TIntHashSet();
    //        userBiases = new TIntDoubleHashMap();
    //        itemBiases = new TIntDoubleHashMap();
            events.stream()
                    .mapToInt(Event::userId)
    //                .peek(user -> userBiases.put(user, DEFAULT_BIAS))
                    .forEach(userIds::add);
            events.stream()
                    .mapToInt(Event::itemId)
    //                .peek(item -> itemBiases.put(item, DEFAULT_BIAS))
                    .forEach(itemIds::add);
            userIdsArray = userIds.toArray();
            itemIdsArray = itemIds.toArray();

            double itemDeltaMean = events.stream()
                    .filter(event -> event.getDelta() >= 0)
                    .collect(Collectors.averagingDouble(Event::getDelta));
            double embMean = sqrt(1 / itemDeltaMean) / dim;
            System.out.println("Embedding mean = " + embMean);
            if (userEmbeddings == null) {
                userEmbeddings = initEmbeddings(userIds, embMean);
            }
            if (itemEmbeddings == null) {
                itemEmbeddings = initEmbeddings(itemIds, embMean);
            }
            dataInitialized = true;
        }
    */
    public static void makeInitialEmbeddings(int dim, List<Event> history,
                                             TIntObjectMap<Vec> userEmbeddings, TIntObjectMap<Vec> itemEmbeddings) {
        final TIntSet userIds = new TIntHashSet();
        final TIntSet itemIds = new TIntHashSet();
        for (Event event : history) {
            userIds.add(event.userId());
            itemIds.add(event.itemId());
        }

        final double itemDeltaMean = DataPreprocessor.groupToSessions(history).stream()
                .mapToDouble(Session::getDelta)
                .filter(delta -> delta >= 0)
                .average().orElse(-1);
        final double embeddingMean = sqrt(1 / itemDeltaMean) / dim;
        final FastRandom randomGenerator = new FastRandom();
        userIds.forEach(userId -> {
            userEmbeddings.put(userId, makeEmbedding(randomGenerator, embeddingMean, dim));
            return true;
        });
        itemIds.forEach(itemId -> {
            itemEmbeddings.put(itemId, makeEmbedding(randomGenerator, embeddingMean, dim));
            return true;
        });
    }

    public void logLikelihoodDerivative(final List<Event> events,
                                        final TIntObjectMap<Vec> userDerivatives,
                                        final TIntObjectMap<Vec> itemDerivatives) {
        final List<Session> sessions = DataPreprocessor.groupToSessions(events);
        final double observationEnd = events.get(sessions.size() - 1).getTs();
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
        for (final Session session : sessions) {
            final long pairId = session.getPair();
            if (!seenPairs.contains(pairId)) {
                seenPairs.add(pairId);
                lambdasByItem.accept(session);
            } else {
                updateDerivativeInnerEvent(lambdasByItem, session.userId(), session.itemId(), session.getDelta(),
                        userDerivatives, itemDerivatives);
                lambdasByItem.accept(session);
                lastVisitTimes.put(pairId, session.getTs());
            }
        }
        for (long pairId : lastVisitTimes.keys()) {
            final int userId = Util.extractUserId(pairId);
            final int itemId = Util.extractItemId(pairId);
            updateDerivativeLastEvent(lambdasByItem, userId, itemId, observationEnd - lastVisitTimes.get(pairId),
                    userDerivatives, itemDerivatives);
        }
    }

    protected void updateDerivativeInnerEvent(LambdaStrategy lambdasByItem, int userId, int itemId, double timeDelta,
                                              TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives) {
        final double lam = lambdasByItem.getLambda(userId, itemId);
        final Vec lamDU = lambdasByItem.getLambdaUserDerivative(userId, itemId);
        final TIntObjectMap<Vec> lamDI = lambdasByItem.getLambdaItemDerivative(userId, itemId);
        final double tLam = lambdaTransform.applyAsDouble(lam);
        final double tau = max(timeDelta, eps);

        final double expPlus = exp(-tLam * (tau + eps));
        final double expMinus = exp(-tLam * (tau - eps));
        final double lamDT = lambdaDerivativeTransform.applyAsDouble(lam);
        final double commonPart = lamDT * ((tau + eps) * expPlus - (tau - eps) * expMinus) / (-expPlus + expMinus);

        if (!Double.isNaN(commonPart)) {
            VecTools.scale(lamDU, commonPart);
            VecTools.append(userDerivatives.get(userId), lamDU);
            for (TIntObjectIterator<Vec> it = lamDI.iterator(); it.hasNext(); ) {
                it.advance();
                final Vec derivative = it.value();
                VecTools.scale(derivative, commonPart);
                VecTools.append(itemDerivatives.get(it.key()), derivative);
            }
        }
    }

    protected void updateDerivativeLastEvent(LambdaStrategy lambdasByItem, int userId, int itemId, double tau,
                                             TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives) {
        final double lam = lambdasByItem.getLambda(userId, itemId);
        final Vec lamDU = lambdasByItem.getLambdaUserDerivative(userId, itemId);
        final TIntObjectMap<Vec> lamDI = lambdasByItem.getLambdaItemDerivative(userId, itemId);
        final double commonPart = -lambdaDerivativeTransform.applyAsDouble(lam) * tau;

        VecTools.scale(lamDU, commonPart);
        VecTools.append(userDerivatives.get(userId), lamDU);
        for (TIntObjectIterator<Vec> it = lamDI.iterator(); it.hasNext(); ) {
            it.advance();
            final Vec derivative = it.value();
            VecTools.scale(derivative, commonPart);
            VecTools.append(itemDerivatives.get(it.key()), derivative);
        }
    }

    public void fit(final List<Event> events, final double learningRate, final int iterationsNumber,
                    final double decay, final FitListener listener) {
        optimizeGD(events, learningRate, iterationsNumber, decay, listener);
    }

    private void optimizeSGD(final List<Event> events, final double learningRate, final int iterationsNumber,
                             final double decay, final FitListener listener) {
        double lr = learningRate / events.size();
        Map<Integer, List<Event>> eventsByUser = new HashMap<>(userIds.size());
        for (final int userId : userIdsArray) {
            eventsByUser.put(userId, new ArrayList<>());
        }
        for (final Event event : events) {
            eventsByUser.get(event.userId()).add(event);
        }

        for (int i = 0; i < iterationsNumber; ++i) {
            for (final List<Event> userEvents : eventsByUser.values()) {
                stepGD(userEvents, lr);
            }
            lr *= decay;
            if (listener != null) {
                listener.apply(this);
            }
        }
    }

    private void optimizeGD(final List<Event> events, final double learningRate, final int iterationsNumber,
                            final double decay, final FitListener listener) {
        double lr = learningRate / events.size();
        for (int i = 0; i < iterationsNumber; ++i) {
            stepGD(events, lr);
            lr *= decay;
            if (listener != null) {
                listener.apply(this);
            }
        }
    }

    private void stepGD(final List<Event> events, double lr) {
        final TIntObjectMap<Vec> userDerivatives = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> itemDerivatives = new TIntObjectHashMap<>();
        logLikelihoodDerivative(events, userDerivatives, itemDerivatives);
        for (final int userId : userIdsArray) {
            Vec userDerivative = userDerivatives.get(userId);
            VecTools.scale(userDerivative, lr);
            VecTools.append(userEmbeddings.get(userId), userDerivative);
        }
        for (final int itemId : itemIdsArray) {
            Vec itemDerivative = itemDerivatives.get(itemId);
            VecTools.scale(itemDerivative, lr);
            VecTools.append(itemEmbeddings.get(itemId), itemDerivative);
        }
    }

    private class ApplicableImpl implements ApplicableModel {
        private final LambdaStrategy lambdaStrategy;

        private ApplicableImpl() {
            lambdaStrategy = lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        }

        @Override
        public void accept(final Session session) {
            lambdaStrategy.accept(session);
        }

        @Override
        public double getLambda(final int userId, final int itemId) {
            return lambdaTransform.applyAsDouble(lambdaStrategy.getLambda(userId, itemId));
        }

        @Override
        public double timeDelta(final int userId, final int itemId) {
            return 1 / getLambda(userId, itemId);
        }

        @Override
        public double probabilityBeforeX(int userId, int itemId, double x) {
            return exp(-getLambda(userId, itemId) * x);
        }
    }

    public ApplicableModel getApplicable(final List<Event> events) {
        ApplicableModel applicable = new ApplicableImpl();
        if (events != null) {
            applicable.fit(events);
        }
        return applicable;
    }

    public ApplicableModel getApplicable() {
        return new ApplicableImpl();
    }

    public TIntObjectMap<Vec> getUserEmbeddings() {
        return userEmbeddings;
    }

    public TIntObjectMap<Vec> getItemEmbeddings() {
        return itemEmbeddings;
    }

    public int getDim() {
        return dim;
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

    private static Map<Integer, Double> biasesToSerializable(final TIntDoubleMap biases) {
        return Arrays.stream(biases.keys())
                .boxed()
                .collect(Collectors.toMap(x -> x, biases::get));
    }

    private static TIntDoubleMap biasesFromSerializable(final Map<Integer, Double> serBiases) {
        final TIntDoubleMap biases = new TIntDoubleHashMap();
        serBiases.forEach(biases::put);
        return biases;
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
//        objectOutputStream.writeObject(biasesToSerializable(userBiases));
        objectOutputStream.writeObject(embeddingsToSerializable(itemEmbeddings));
//        objectOutputStream.writeObject(biasesToSerializable(itemEmbeddings));
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
//        final TIntDoubleMap userBiases =
//                biasesFromSerializable((Map<Integer, Double>) objectInputStream.readObject());
        final TIntObjectMap<Vec> itemEmbeddings =
                embeddingsFromSerializable((Map<Integer, double[]>) objectInputStream.readObject());
//        final TIntDoubleMap itemBiases =
//                biasesFromSerializable((Map<Integer, Double>) objectInputStream.readObject());
        return new Model(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform,
                lambdaStrategyFactory, userEmbeddings, itemEmbeddings);
//                lambdaStrategyFactory, userEmbeddings, itemEmbeddings);
    }

    public interface FitListener {
        void apply(Model model);
    }
}
