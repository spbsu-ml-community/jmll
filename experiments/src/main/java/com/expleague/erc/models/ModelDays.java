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
        initIds();
        userDayAvgStarts = calcAvgStarts(events);
        averageOneDayDelta = calcAverageOneDayDelta(events);
        userDayBorders = new TIntIntHashMap();
        userDayPeaks = new TIntIntHashMap();
        calcDayPoints(events, userDayBorders, userDayPeaks);
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

    public static void calcDayPoints(final List<Event> events, final TIntIntMap userDayBorders, final TIntIntMap userDayPeaks) {
        final TIntObjectMap<long[]> counters = new TIntObjectHashMap<>();
        for (final Event event : events) {
            final int userId = event.userId();
            if (!counters.containsKey(userId)) {
                counters.put(userId, new long[DAY_HOURS]);
            }
            counters.get(userId)[(int) event.getTs() % DAY_HOURS]++;
        }
        counters.forEachEntry((userId, userCounters) -> {
            int argMax = -1;
            int argMin = -1;
            long minCounter = Long.MAX_VALUE;
            long maxCounter = Long.MIN_VALUE;
            for (int i = 0; i < DAY_HOURS; i++) {
                if (userCounters[i] < minCounter) {
                    minCounter = userCounters[i];
                    argMin = i;
                }
                if (userCounters[i] > maxCounter) {
                    maxCounter = userCounters[i];
                    argMax = i;
                }
            }
            userDayBorders.put(userId, argMin);
            userDayPeaks.put(userId, argMax);
            return true;
        });
    }

    private static TIntDoubleMap calcAvgStarts(final List<Event> events) {
        final TIntIntMap lastDays = new TIntIntHashMap();
        TIntDoubleMap starts = new TIntDoubleHashMap();
        TIntIntMap count = new TIntIntHashMap();
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int userId = session.userId();
            final int curDay = ((int) session.getStartTs() / DAY_HOURS);
            final double curTime = session.getStartTs() - curDay * DAY_HOURS;
            if (!lastDays.containsKey(userId) || lastDays.get(userId) != curDay) {
                starts.adjustOrPutValue(userId, curTime, curTime);
                count.adjustOrPutValue(userId, 1, 1);
                lastDays.put(userId, curDay);
            }
        }
        TIntDoubleMap avgStarts = new TIntDoubleHashMap();
        for (final int userId : starts.keys()) {
            avgStarts.put(userId, starts.get(userId) / count.get(userId));
        }
        return avgStarts;
    }

    private static TIntDoubleMap calcAverageOneDayDelta(final List<Event> events) {
        final TIntIntMap lastDays = new TIntIntHashMap();
        final TIntDoubleMap lastDayTimes = new TIntDoubleHashMap();
        final TIntDoubleMap userDeltas = new TIntDoubleHashMap();
        final TIntIntMap userCounts = new TIntIntHashMap();
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int userId = session.userId();
            final int curDay = ((int) session.getStartTs() / DAY_HOURS);
            final double curTime = session.getStartTs() - curDay * DAY_HOURS;
            if (!lastDays.containsKey(userId)) {
                lastDays.put(userId, curDay);
                lastDayTimes.put(userId, curTime);
                continue;
            }
            final int lastDay = lastDays.get(userId);
            if (lastDay == curDay) {
                final double delta = curTime - lastDayTimes.get(userId);
                userDeltas.adjustOrPutValue(userId, delta, delta);
                userCounts.adjustOrPutValue(userId, 1, 1);
                lastDayTimes.put(userId, curTime);
            } else {
                lastDays.put(userId, curDay);
                lastDayTimes.put(userId, curTime);
            }
        }
        final TIntDoubleMap result = new TIntDoubleHashMap();
        for (final int userId : userDeltas.keys()) {
            result.put(userId, userDeltas.get(userId) / userCounts.get(userId));
        }
        return result;
    }

    @Override
    protected void write(final ObjectOutputStream objectOutputStream) throws IOException {
        objectOutputStream.writeInt(dim);
        objectOutputStream.writeDouble(beta);
        objectOutputStream.writeDouble(eps);
        objectOutputStream.writeDouble(otherItemImportance);
        objectOutputStream.writeObject(lambdaTransform);
        objectOutputStream.writeObject(lambdaDerivativeTransform);
        objectOutputStream.writeObject(Util.embeddingsToSerializable(userEmbeddings));
        objectOutputStream.writeObject(Util.embeddingsToSerializable(itemEmbeddings));
        objectOutputStream.writeObject(Util.intDoubleMapToSerializable(initialLambdas));
        objectOutputStream.writeObject(timeTransform);
        objectOutputStream.writeDouble(lowerRangeBorder);
        objectOutputStream.writeDouble(higherRangeBorder);
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
