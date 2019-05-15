package com.expleague.erc.models;

import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.erc.*;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TLongDoubleHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.TLongSet;
import gnu.trove.set.hash.TIntHashSet;
import gnu.trove.set.hash.TLongHashSet;

import java.io.*;
import java.util.*;
import java.util.function.DoubleUnaryOperator;

import static java.lang.Math.*;

public class Model implements Serializable {

    protected final int dim;
    protected final double beta;
    protected final double eps;
    protected final double otherItemImportance;
    protected final DoubleUnaryOperator lambdaTransform;
    protected final DoubleUnaryOperator lambdaDerivativeTransform;
    protected final LambdaStrategyFactory lambdaStrategyFactory;
    protected TIntObjectMap<Vec> userEmbeddings;
    protected TIntObjectMap<Vec> itemEmbeddings;
    protected TIntDoubleMap initialLambdas;
    protected TIntSet userIds;
    protected TIntSet itemIds;
    protected int[] userIdsArray;
    protected int[] itemIdsArray;
    protected boolean isInit = false;
    protected final TimeTransformer timeTransform;
    protected final double lowerRangeBorder;
    protected final double higherRangeBorder;

    public Model(final int dim, final double beta, final double eps, final double otherItemImportance,
                 final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                 final LambdaStrategyFactory lambdaStrategyFactory) {
        this(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                new TIntObjectHashMap<>(), new TIntObjectHashMap<>(), null,
                new Util.GetDelta(), Util.MAX_GAP, Util.CHURN_THRESHOLD);
    }

    public Model(final int dim, final double beta, final double eps, final double otherItemImportance,
                 final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                 final LambdaStrategyFactory lambdaStrategyFactory, final TimeTransformer timeTransform,
                 final double lowerRangeBorder, final double higherRangeBorder) {
        this(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                new TIntObjectHashMap<>(), new TIntObjectHashMap<>(), null, timeTransform,
                lowerRangeBorder, higherRangeBorder);
    }

    public Model(final int dim, final double beta, final double eps, final double otherItemImportance,
                 final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                 final LambdaStrategyFactory lambdaStrategyFactory, final TIntObjectMap<Vec> usersEmbeddingsPrior,
                 final TIntObjectMap<Vec> itemsEmbeddingsPrior, final TIntDoubleMap initialLambdas) {
        this(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                usersEmbeddingsPrior, itemsEmbeddingsPrior, initialLambdas, new Util.GetDelta(),
                Util.MAX_GAP, Util.CHURN_THRESHOLD);
    }

    public Model(final int dim, final double beta, final double eps, final double otherItemImportance,
                 final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                 final LambdaStrategyFactory lambdaStrategyFactory, final TIntObjectMap<Vec> usersEmbeddingsPrior,
                 final TIntObjectMap<Vec> itemsEmbeddingsPrior, final TIntDoubleMap initialLambdas,
                 final TimeTransformer timeTransform, final double lowerRangeBorder, final double higherRangeBorder) {
        this.dim = dim;
        this.beta = beta;
        this.eps = eps;
        this.otherItemImportance = otherItemImportance;
        this.lambdaTransform = lambdaTransform;
        this.lambdaDerivativeTransform = lambdaDerivativeTransform;
        this.lambdaStrategyFactory = lambdaStrategyFactory;
        if (usersEmbeddingsPrior != null) {
            userEmbeddings = usersEmbeddingsPrior;
        } else {
            userEmbeddings = new TIntObjectHashMap<>();
        }
        userIds = userEmbeddings.keySet();
        userIdsArray = userIds.toArray();
        if (itemsEmbeddingsPrior != null) {
            itemEmbeddings = itemsEmbeddingsPrior;
        } else {
            itemEmbeddings = new TIntObjectHashMap<>();
        }
        itemIds = itemEmbeddings.keySet();
        itemIdsArray = itemIds.toArray();
        this.initialLambdas = initialLambdas;

        this.timeTransform = timeTransform;
        this.lowerRangeBorder = lowerRangeBorder;
        this.higherRangeBorder = higherRangeBorder;
    }

    // Init

    public void initModel(final List<Event> events) {
        if (isInit) {
            return;
        }
        makeInitialEmbeddings(events);
        initIds();
        isInit = true;
    }

    // If the embeddings are already present
    public void initModel() {
        initIds();
        isInit = true;
    }

    protected void initIds() {
        userIds = userEmbeddings.keySet();
        userIdsArray = userIds.toArray();
        itemIds = itemEmbeddings.keySet();
        itemIdsArray = itemIds.toArray();
    }

    protected static Vec getGaussianEmbedding(final FastRandom randomGenerator, final double embMean, final int dim) {
        Vec embedding = new ArrayVec(dim);
        VecTools.fillGaussian(embedding, randomGenerator);
        VecTools.scale(embedding, embMean / 2);
        VecTools.adjust(embedding, embMean);
        embedding = VecTools.abs(embedding);
        return embedding;
    }

    protected static Vec getUniformEmbedding(final FastRandom randomGenerator, final double from, final double to, final int dim) {
        Vec embedding = new ArrayVec(dim);
        VecTools.adjust(VecTools.fillUniform(embedding, randomGenerator, (to - from) / 2), (to + from) / 2);
        return embedding;
    }

    protected void makeInitialEmbeddings(List<Event> history) {
        userIds = new TIntHashSet();
        itemIds = new TIntHashSet();
        for (final Event event : history) {
            userIds.add(event.userId());
            itemIds.add(event.itemId());
        }

        final FastRandom randomGenerator = new FastRandom();
        final double edge = 0.;
        userIds.forEach(userId -> {
            userEmbeddings.put(userId, getUniformEmbedding(randomGenerator, -edge, edge, dim));
//            VecTools.normalizeL2(userEmbeddings.get(userId));
            return true;
        });
        itemIds.forEach(itemId -> {
            itemEmbeddings.put(itemId, getUniformEmbedding(randomGenerator, -edge, edge, dim));
//            VecTools.normalizeL2(itemEmbeddings.get(itemId));
            return true;
        });
    }

    // LogLikelihood

    public void logLikelihoodDerivative(final List<Event> events,
                                        final TIntObjectMap<Vec> userDerivatives,
                                        final TIntObjectMap<Vec> itemDerivatives,
                                        final TIntDoubleMap initialLambdasDerivatives) {
        final List<EventSeq> eventSeqs = DataPreprocessor.groupToEventSeqs(events);
        final double observationEnd = events.get(events.size() - 1).getTs();
        final TLongSet seenPairs = new TLongHashSet();
        fillInitDerivatives(userDerivatives, itemDerivatives);
        final LambdaStrategy lambdasByItem =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, initialLambdas, beta, otherItemImportance);
        final TLongDoubleMap lastVisitTimes = new TLongDoubleHashMap();
        for (final EventSeq eventSeq : eventSeqs) {
            final long pairId = eventSeq.getPair();
            if (!seenPairs.contains(pairId)) {
                seenPairs.add(pairId);
                lambdasByItem.accept(eventSeq);
            } else {
                updateDerivativeInnerEvent(lambdasByItem, eventSeq.userId(), eventSeq.itemId(),
                        timeTransform.apply(eventSeq.getDelta(), eventSeq.getStartTs(), eventSeq.userId()),
                        userDerivatives, itemDerivatives, initialLambdasDerivatives);
                lambdasByItem.accept(eventSeq);
                lastVisitTimes.put(pairId, eventSeq.getStartTs());
            }
        }
        for (final long pairId : lastVisitTimes.keys()) {
            final int userId = Util.extractUserId(pairId);
            final int itemId = Util.extractItemId(pairId);
            updateDerivativeLastEvent(lambdasByItem, userId, itemId,
                    timeTransform.apply(observationEnd - lastVisitTimes.get(pairId), observationEnd, userId),
                    userDerivatives, itemDerivatives);
        }
    }

    protected void updateDerivativeInnerEvent(final LambdaStrategy lambdasByItem, final int userId, final int itemId,
                                              final double timeDelta, final TIntObjectMap<Vec> userDerivatives,
                                              final TIntObjectMap<Vec> itemDerivatives,
                                              final TIntDoubleMap initialLambdasDerivatives) {
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

    protected void fillInitDerivatives(final TIntObjectMap<Vec> userDerivatives,
                                       final TIntObjectMap<Vec> itemDerivatives) {
        for (final int userId : userIdsArray) {
            userDerivatives.put(userId, new ArrayVec(dim));
        }
        for (final int itemId : itemIdsArray) {
            itemDerivatives.put(itemId, new ArrayVec(dim));
        }
    }

    // Fit

    public void fit(final List<Event> events, final double learningRate, final int iterationsNumber,
                    final double decay, final FitListener listener) {
        initModel(events);
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

        for (int i = 0; i < iterationsNumber; i++) {
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
        if (listener != null) {
            listener.apply(this);
        }
        for (int i = 0; i < iterationsNumber; i++) {
            stepGD(events, lr);
            lr *= decay;
            if (listener != null) {
                listener.apply(this);
            }
        }
    }

    protected void stepGD(final List<Event> events, double lr) {
        final TIntObjectMap<Vec> userDerivatives = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> itemDerivatives = new TIntObjectHashMap<>();
        logLikelihoodDerivative(events, userDerivatives, itemDerivatives, null);
        for (final int userId : userIdsArray) {
            Vec userDerivative = userDerivatives.get(userId);
            VecTools.scale(userDerivative, lr);
            Vec userEmbedding = userEmbeddings.get(userId);
            VecTools.append(userEmbedding, userDerivative);
//            VecTools.normalizeL2(userEmbedding);
        }
        for (final int itemId : itemIdsArray) {
            Vec itemDerivative = itemDerivatives.get(itemId);
            VecTools.scale(itemDerivative, lr);
            Vec itemEmbedding = itemEmbeddings.get(itemId);
            VecTools.append(itemEmbedding, itemDerivative);
//            VecTools.normalizeL2(itemEmbedding);
        }
    }

    // Applicable

    private class ApplicableImpl implements ApplicableModel {
        private final LambdaStrategy lambdaStrategy;

        private ApplicableImpl() {
            lambdaStrategy = lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, initialLambdas, beta, otherItemImportance);
        }

        @Override
        public void accept(final EventSeq eventSeq) {
            lambdaStrategy.accept(eventSeq);
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
            return 1 - exp(-getLambda(userId, itemId) * x);
        }
    }

    public ApplicableModel getApplicable(final List<Event> events) {
        initModel(events);
        ApplicableModel applicable = getApplicable();
        if (events != null) {
            applicable.fit(events);
        }
        return applicable;
    }

    public ApplicableModel getApplicable() {
        if (!isInit) {
            throw new IllegalStateException("Model is not initialized");
        }
        return new ApplicableImpl();
    }

    // Getters

    public TIntObjectMap<Vec> getUserEmbeddings() {
        return userEmbeddings;
    }

    public TIntObjectMap<Vec> getItemEmbeddings() {
        return itemEmbeddings;
    }

    public int getDim() {
        return dim;
    }

    public boolean forPrediction(final Session session) {
        final double delta = timeTransform.apply(session);
        return lowerRangeBorder <= delta && delta <= higherRangeBorder;
    }

    // Save & Load

    public void write(final OutputStream stream) throws IOException {
        final ObjectOutputStream objectOutputStream = new ObjectOutputStream(stream);
        objectOutputStream.writeInt(dim);
        objectOutputStream.writeDouble(beta);
        objectOutputStream.writeDouble(eps);
        objectOutputStream.writeDouble(otherItemImportance);
        objectOutputStream.writeObject(lambdaTransform);
        objectOutputStream.writeObject(lambdaDerivativeTransform);
        objectOutputStream.writeObject(lambdaStrategyFactory);
        objectOutputStream.writeObject(Util.embeddingsToSerializable(userEmbeddings));
        objectOutputStream.writeObject(Util.embeddingsToSerializable(itemEmbeddings));
        objectOutputStream.writeObject(Util.intDoubleMapToSerializable(initialLambdas));
        objectOutputStream.writeObject(timeTransform);
        objectOutputStream.writeDouble(lowerRangeBorder);
        objectOutputStream.writeDouble(higherRangeBorder);
        objectOutputStream.close();
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
                Util.embeddingsFromSerializable((Map<Integer, double[]>) objectInputStream.readObject());
        final TIntObjectMap<Vec> itemEmbeddings =
                Util.embeddingsFromSerializable((Map<Integer, double[]>) objectInputStream.readObject());
        final TimeTransformer timeTransform = (TimeTransformer) objectInputStream.readObject();
        final TIntDoubleMap initialLambdas =
                Util.intDoubleMapFromSerializable((Map<Integer, Double>) objectInputStream.readObject());
        final double lowerRangeBorder = objectInputStream.readDouble();
        final double higherRangeBorder = objectInputStream.readDouble();
        final Model model = new Model(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform,
                lambdaStrategyFactory, userEmbeddings, itemEmbeddings, initialLambdas,
                timeTransform, lowerRangeBorder, higherRangeBorder);
        model.initModel();
        return model;
    }

    public interface FitListener {
        void apply(Model model);
    }
}
