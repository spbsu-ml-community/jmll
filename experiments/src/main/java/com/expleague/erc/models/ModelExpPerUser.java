package com.expleague.erc.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.iterator.TIntObjectIterator;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.io.*;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Predicate;

import static java.lang.Math.exp;
import static java.lang.Math.max;

public class ModelExpPerUser extends Model {
    protected final TIntDoubleMap initialLambdas;

    public ModelExpPerUser(int dim, double beta, double eps, double otherItemImportance, DoubleUnaryOperator lambdaTransform,
                           DoubleUnaryOperator lambdaDerivativeTransform, LambdaStrategyFactory lambdaStrategyFactory,
                           TIntDoubleMap initialLambdas) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
        this.initialLambdas = initialLambdas;
    }

    public ModelExpPerUser(final int dim, final double beta, final double eps, final double otherItemImportance,
                           final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                           final LambdaStrategyFactory lambdaStrategyFactory, TIntDoubleMap initialLambdas,
                           final BiFunction<Double, Integer, Double> timeTransform, final double lowerRangeBorder,
                           final double higherRangeBorder) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                new TIntObjectHashMap<>(), new TIntObjectHashMap<>(), timeTransform, lowerRangeBorder, higherRangeBorder);
        this.initialLambdas = initialLambdas;
    }

    public ModelExpPerUser(int dim, double beta, double eps, double otherItemImportance,
                           DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                           LambdaStrategyFactory lambdaStrategyFactory, TIntDoubleMap initialLambdas,
                           TIntObjectMap<Vec> usersEmbeddingsPrior, TIntObjectMap<Vec> itemsEmbeddingsPrior,
                           BiFunction<Double, Integer, Double> timeTransform,
                           double lowerRangeBorder, double higherRangeBorder) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                usersEmbeddingsPrior, itemsEmbeddingsPrior, timeTransform, lowerRangeBorder, higherRangeBorder);
        this.initialLambdas = initialLambdas;
    }

    @Override
    public void initModel(final List<Event> events) {
        if (isInit) {
            return;
        }
        makeInitialEmbeddings(events);
        initIds();
        isInit = true;
    }

    @Override
    public void logLikelihoodDerivative(final List<Event> events,
                                        final TIntObjectMap<Vec> userDerivatives,
                                        final TIntObjectMap<Vec> itemDerivatives,
                                        final TIntDoubleMap initialLambdasDerivatives) {
        fillInitDerivatives(userDerivatives, itemDerivatives);
        final LambdaStrategy lambdaStrategy =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            if (forPrediction(session)) {
                updateDerivativeInnerEvent(lambdaStrategy, session.userId(),
                        timeTransform.apply(session.getDelta(), session.userId()),
                        userDerivatives, itemDerivatives, initialLambdasDerivatives);
            }
            session.getEventSeqs().forEach(lambdaStrategy::accept);
        }
    }

    protected void updateDerivativeInnerEvent(LambdaStrategy lambdasByItem, int userId, double timeDelta,
                                              TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives,
                                              TIntDoubleMap initialLambdasDerivatives) {
        final double lam = lambdasByItem.getLambda(userId);
        final Vec lamDU = lambdasByItem.getLambdaUserDerivative(userId);
        final TIntObjectMap<Vec> lamDI = lambdasByItem.getLambdaItemDerivative(userId);
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
            initialLambdasDerivatives.adjustOrPutValue(userId, commonPart, commonPart);
        }
    }

    @Override
    protected void stepGD(final List<Event> events, double lr) {
        final TIntObjectMap<Vec> userDerivatives = new TIntObjectHashMap<>();
        final TIntObjectMap<Vec> itemDerivatives = new TIntObjectHashMap<>();
        final TIntDoubleMap initialLambdaDerivatives = new TIntDoubleHashMap();
        logLikelihoodDerivative(events, userDerivatives, itemDerivatives, initialLambdaDerivatives);
        for (final int userId : userIdsArray) {
            Vec userDerivative = userDerivatives.get(userId);
            VecTools.scale(userDerivative, lr);
            Vec userEmbedding = userEmbeddings.get(userId);
            VecTools.append(userEmbedding, userDerivative);
//            VecTools.normalizeL2(userEmbedding);
            initialLambdas.adjustValue(userId, initialLambdaDerivatives.get(userId) * lr);
        }
        for (final int itemId : itemIdsArray) {
            Vec itemDerivative = itemDerivatives.get(itemId);
            VecTools.scale(itemDerivative, lr);
            Vec itemEmbedding = itemEmbeddings.get(itemId);
            VecTools.append(itemEmbedding, itemDerivative);
//            VecTools.normalizeL2(itemEmbedding);
        }
    }

    private class ApplicableImpl implements ApplicableModel {
        private final LambdaStrategy lambdaStrategy;

        private ApplicableImpl() {
            lambdaStrategy = lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, beta, otherItemImportance);
        }

        @Override
        public void accept(final EventSeq eventSeq) {
            lambdaStrategy.accept(eventSeq);
        }

        @Override
        public double getLambda(int userId) {
            return lambdaTransform.applyAsDouble(lambdaStrategy.getLambda(userId));
        }

        @Override
        public double timeDelta(final int userId, final double time) {
            return 1 / getLambda(userId);
        }

        @Override
        public double probabilityBeforeX(int userId, double x) {
            return 1 - exp(-getLambda(userId) * x);
        }

        @Override
        public double probabilityBeforeX(int userId, int itemId, double x) {
            return 1 - exp(-getLambda(userId, itemId) * x);
        }
    }

    @Override
    public ApplicableModel getApplicable() {
        return new ApplicableImpl();
    }

    @Override
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

    public static ModelExpPerUser load(final InputStream stream) throws IOException, ClassNotFoundException {
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
        final TIntDoubleMap initialLambdas =
                Util.intDoubleMapFromSerializable((Map<Integer, Double>) objectInputStream.readObject());
        final BiFunction<Double, Integer, Double> timeTransform = (BiFunction<Double, Integer, Double>) objectInputStream.readObject();
        final double lowerRangeBorder = objectInputStream.readDouble();
        final double higherRangeBorder = objectInputStream.readDouble();
        final ModelExpPerUser model = new ModelExpPerUser(dim, beta, eps, otherItemImportance, lambdaTransform,
                lambdaDerivativeTransform, lambdaStrategyFactory, initialLambdas, userEmbeddings, itemEmbeddings,
                timeTransform, lowerRangeBorder, higherRangeBorder);
        model.initModel();
        return model;
    }
}
