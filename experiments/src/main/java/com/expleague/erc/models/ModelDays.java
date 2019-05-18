package com.expleague.erc.models;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.*;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategy;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import com.expleague.erc.lambda.PerUserLambdaStrategy;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.DoubleUnaryOperator;

import static com.expleague.erc.Util.DAY_HOURS;
import static java.lang.Math.*;

public class ModelDays extends ModelExpPerUser {
    private TIntIntMap userDayBorders;
    private TIntIntMap userDayPeaks;
    private TIntDoubleMap userDayAvgStarts;
    private TIntDoubleMap averageOneDayDelta;

    public ModelDays(int dim, double beta, double eps, double otherItemImportance, DoubleUnaryOperator lambdaTransform,
                     DoubleUnaryOperator lambdaDerivativeTransform, LambdaStrategyFactory lambdaStrategyFactory) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
    }

    public ModelDays(final int dim, final double beta, final double eps, final double otherItemImportance,
                     final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                     final LambdaStrategyFactory lambdaStrategyFactory, TIntDoubleMap initialLambdas,
                     final TimeTransformer timeTransform, final double lowerRangeBorder,
                     final double higherRangeBorder) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                initialLambdas, new TIntObjectHashMap<>(), new TIntObjectHashMap<>(), timeTransform,
                lowerRangeBorder, higherRangeBorder);
    }

    public ModelDays(int dim, double beta, double eps, double otherItemImportance,
                     final DoubleUnaryOperator lambdaTransform, final DoubleUnaryOperator lambdaDerivativeTransform,
                     final LambdaStrategyFactory lambdaStrategyFactory, final TIntDoubleMap initialLambdas,
                     final TIntObjectMap<Vec> usersEmbeddingsPrior, final TIntObjectMap<Vec> itemsEmbeddingsPrior,
                     final TIntIntMap userDayBorders, TIntIntMap userDayPeaks, final TIntDoubleMap userDayAvgStarts,
                     final TIntDoubleMap averageOneDayDelta, final TimeTransformer timeTransform,
                     final double lowerRangeBorder, final double higherRangeBorder) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory,
                initialLambdas, usersEmbeddingsPrior, itemsEmbeddingsPrior, timeTransform,
                lowerRangeBorder, higherRangeBorder);
        this.userDayBorders = userDayBorders;
        this.userDayAvgStarts = userDayAvgStarts;
        this.averageOneDayDelta = averageOneDayDelta;
        this.userDayPeaks = userDayPeaks;
    }

    @Override
    public void initModel(final List<Event> events) {
        if (isInit) {
            return;
        }
        makeInitialEmbeddings(events);
        makeInitialLambdas(events);
        initIds();
        averageOneDayDelta = ConstantNextTimeModel.calcAverageOneDayDelta(events);
        userDayBorders = new TIntIntHashMap();
        userDayPeaks = new TIntIntHashMap();
        ModelCombined.calcDayPoints(events, userDayBorders, userDayPeaks);
        userDayAvgStarts = ModelCombined.calcAvgStarts(events, userDayBorders);
        isInit = true;
    }

    @Override
    public void logLikelihoodDerivative(List<Event> events,
                                        TIntObjectMap<Vec> userDerivatives, TIntObjectMap<Vec> itemDerivatives,
                                        TIntDoubleMap initialLambdasDerivatives) {
        fillInitDerivatives(userDerivatives, itemDerivatives);
        final LambdaStrategy lambdaStrategy =
                lambdaStrategyFactory.get(userEmbeddings, itemEmbeddings, initialLambdas, beta, otherItemImportance);
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            if (!Util.isShortSession(session.getDelta()) && !Util.isDead(session.getDelta())) {
                final int daysPassed = Util.getDaysFromPrevSession(session, userDayBorders.get(session.userId()));
                updateDerivativeInnerEvent(lambdaStrategy, session.userId(), daysPassed, userDerivatives,
                        itemDerivatives, initialLambdasDerivatives);
            }
            session.getEventSeqs().forEach(lambdaStrategy::accept);
        }
    }

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
        public double getLambda(int userId) {
            return lambdaTransform.applyAsDouble(lambdaStrategy.getLambda(userId));
        }

        @Override
        public double timeDelta(final int userId, final double time) {
            final double rawPrediction = 1 / getLambda(userId);
//            final long daysPrediction = round(rawPrediction);
            final int daysPrediction = (int) rawPrediction;
            if (time + rawPrediction * DAY_HOURS < Util.getDay(time, userDayBorders.get(userId)) + DAY_HOURS) {
                return averageOneDayDelta.get(userId);
            }
            final int predictedDay = (int) (time + daysPrediction * DAY_HOURS) / DAY_HOURS;
            double predictedTime = predictedDay * DAY_HOURS + userDayAvgStarts.get(userId);
//            if (userDayAvgStarts.get(userId) < userDayBorders.get(userId)) {
//                predictedTime += DAY_HOURS;
//            }
            if (predictedTime < time) {
                return averageOneDayDelta.get(userId);
            }
            return predictedTime - time;
        }

        @Override
        public double probabilityBeforeX(int userId, double x) {
            return 1 - exp(-getLambda(userId) * x);
        }
    }

    public ApplicableModel getApplicable() {
        return new ApplicableImpl();
    }

    @Override
    protected void write(final ObjectOutputStream objectOutputStream) throws IOException {
        writeBase(objectOutputStream);
        writeLearnedParams(objectOutputStream);
        writeTimeParams(objectOutputStream);

        objectOutputStream.writeObject(Util.intIntMapToSerializable(userDayBorders));
        objectOutputStream.writeObject(Util.intIntMapToSerializable(userDayPeaks));
        objectOutputStream.writeObject(Util.intDoubleMapToSerializable(userDayAvgStarts));
        objectOutputStream.writeObject(Util.intDoubleMapToSerializable(averageOneDayDelta));
    }

    public static ModelDays load(final Path path) throws IOException, ClassNotFoundException {
        final ObjectInputStream objectInputStream = new ObjectInputStream(Files.newInputStream(path));
        final ModelDays model = read(objectInputStream);
        objectInputStream.close();
        return model;
    }

    public static ModelDays read(final ObjectInputStream objectInputStream) throws IOException, ClassNotFoundException {
        final int dim = objectInputStream.readInt();
        final double beta = objectInputStream.readDouble();
        final double eps = objectInputStream.readDouble();
        final double otherItemImportance = objectInputStream.readDouble();
        final DoubleUnaryOperator lambdaTransform = (DoubleUnaryOperator) objectInputStream.readObject();
        final DoubleUnaryOperator lambdaDerivativeTransform = (DoubleUnaryOperator) objectInputStream.readObject();
        final TIntObjectMap<Vec> userEmbeddings =
                Util.embeddingsFromSerializable((Map<Integer, double[]>) objectInputStream.readObject());
        final TIntObjectMap<Vec> itemEmbeddings =
                Util.embeddingsFromSerializable((Map<Integer, double[]>) objectInputStream.readObject());
        final TIntDoubleMap initialLambdas =
                Util.intDoubleMapFromSerializable((Map<Integer, Double>) objectInputStream.readObject());
        final LambdaStrategyFactory lambdaStrategyFactory = new PerUserLambdaStrategy.Factory();
        final TimeTransformer timeTransform = (TimeTransformer) objectInputStream.readObject();
        final double lowerRangeBorder = objectInputStream.readDouble();
        final double higherRangeBorder = objectInputStream.readDouble();
        final TIntIntMap userDayBorders =
                Util.intIntMapFromSerializable((Map<Integer, Integer>) objectInputStream.readObject());
        final TIntIntMap userDayPeaks =
                Util.intIntMapFromSerializable((Map<Integer, Integer>) objectInputStream.readObject());
        final TIntDoubleMap userAverageDayStarts =
                Util.intDoubleMapFromSerializable((Map<Integer, Double>) objectInputStream.readObject());
        final TIntDoubleMap averageOneDayDelta =
                Util.intDoubleMapFromSerializable((Map<Integer, Double>) objectInputStream.readObject());
        final ModelDays model = new ModelDays(dim, beta, eps, otherItemImportance, lambdaTransform,
                lambdaDerivativeTransform, lambdaStrategyFactory, initialLambdas, userEmbeddings, itemEmbeddings,
                userDayBorders, userDayPeaks, userAverageDayStarts, averageOneDayDelta, timeTransform,
                lowerRangeBorder, higherRangeBorder);
        model.initModel();
        return model;
    }
}
