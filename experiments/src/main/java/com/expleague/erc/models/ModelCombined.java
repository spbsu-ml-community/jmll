package com.expleague.erc.models;

import com.expleague.erc.*;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TLongDoubleMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TLongDoubleHashMap;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

import static com.expleague.erc.Util.DAY_HOURS;

public class ModelCombined extends Model {
    private final TIntIntMap userDayBorders;
    private final TIntIntMap userDayPeaks;
    private TIntDoubleMap userDayAvgStarts;
    private TLongDoubleMap userDayAverageDeltas;

    private Model daysModel;
    private Model timeModel;

    public static class DayExtractor implements TimeTransformer, Serializable {
        private final TIntIntMap userDayBorders;

        public DayExtractor(TIntIntMap userDayBorders) {
            this.userDayBorders = userDayBorders;
        }

        @Override
        public double apply(double delta, double time, int userId) {
            return Util.getDaysFromPrevSession(delta, time, userDayBorders.get(userId));
        }
    }

    public ModelCombined(int dim, double beta, double eps, double otherItemImportance,
                         DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                         LambdaStrategyFactory lambdaStrategyFactory) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
        userDayBorders = new TIntIntHashMap();
        userDayPeaks = new TIntIntHashMap();
        daysModel = new ModelExpPerUser(dim, beta, eps / 10, otherItemImportance, lambdaTransform, lambdaDerivativeTransform,
                lambdaStrategyFactory, new DayExtractor(userDayBorders), Double.NEGATIVE_INFINITY, Util.CHURN_THRESHOLD_DAYS);
        timeModel = new ConstantNextTimeModel(dim, beta, eps, otherItemImportance, lambdaTransform,
                lambdaDerivativeTransform, lambdaStrategyFactory);
    }

    private ModelCombined(int dim, double beta, double eps, double otherItemImportance,
                          DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                          LambdaStrategyFactory lambdaStrategyFactory, Model daysModel, Model timeModel,
                          TIntIntMap userDayBorders, TIntIntMap userDayPeaks, TIntDoubleMap userDayAvgStarts,
                          TLongDoubleMap userDayAverageDeltas) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
        this.daysModel = daysModel;
        this.timeModel = timeModel;
        this.userDayBorders = userDayBorders;
        this.userDayPeaks = userDayPeaks;
        this.userDayAvgStarts = userDayAvgStarts;
        this.userDayAverageDeltas = userDayAverageDeltas;
    }

    @Override
    public void initModel(final List<Event> events) {
        if (isInit) {
            return;
        }
        daysModel.makeInitialEmbeddings(events);
        timeModel.makeInitialEmbeddings(events);
        initIds();
        calcDayPoints(events, userDayBorders, userDayPeaks);
        userDayAvgStarts = calcAvgStarts(events, userDayBorders);
        calcDayAverages(events);
        isInit = true;
    }

    @Override
    public void initModel() {
        initIds();
        isInit = true;
    }

    @Override
    public void fit(final List<Event> events, final double learningRate, final int iterationsNumber,
                    final double decay) {
        initModel(events);
        daysModel.initModel(events);
        timeModel.initModel(events);
        if (fitListener != null) {
            fitListener.apply(this);
        }
//        final Random random = new Random();
//        final int batchSize = events.size() / 20;
        for (int i = 0; i < iterationsNumber; i++) {
//            final int start = Math.abs(random.nextInt()) % (events.size() - batchSize);
            double lr = learningRate / Math.log(2 + i);
            daysModel.fit(events, lr, 1, 1);
            timeModel.fit(events, lr, 1, 1);
//            daysModel.fit(events.subList(start, start + batchSize), lr, 1, 1);
//            timeModel.fit(events.subList(start, start + batchSize), lr, 1, 1);
            if (fitListener != null) {
                fitListener.apply(this);
            }
            System.out.println();
        }
    }

    public void setDaysFitListener(FitListener listener) {
        daysModel.setFitListener(listener);
    }

    public void setTimeFitListener(FitListener listener) {
        timeModel.setFitListener(listener);
    }

    private class ApplicableImpl implements ApplicableModel {
        private final ApplicableModel daysApplicable;
        private final ApplicableModel timeApplicable;

        private ApplicableImpl(final ApplicableModel daysApplicable, final ApplicableModel timeApplicable) {
            this.daysApplicable = daysApplicable;
            this.timeApplicable = timeApplicable;
        }

        @Override
        public void accept(final EventSeq eventSeq) {
            daysApplicable.accept(eventSeq);
            timeApplicable.accept(eventSeq);
        }

        /**
         * Calculates the estimated exact time before the next session that is expected to happen in a given number of days
         * @param userId the user for whom the prediction is being made
         * @param time time when the prediction is made
         * @param daysStep the number of days to jump
         * @return the exact expected time to the next session
         */
        private double daysPredictionToExact(int userId, double time, int daysStep) {
            final long userDay = combine(userId, daysStep);
            if (userDayAverageDeltas.containsKey(userDay)) {
                return userDayAverageDeltas.get(userDay);
            }
            final int predictedDay = (int) (time + daysStep * DAY_HOURS) / DAY_HOURS;
            final double predictedTime = predictedDay * DAY_HOURS + userDayAvgStarts.get(userId);
            return predictedTime - time;
        }

        @Override
        public double timeDelta(final int userId, final double time) {
            return expectedTime(userId, time);
//            final int daysPrediction = (int) daysApplicable.timeDelta(userId, time);
//            if (daysPrediction == 0) {
//                return timeApplicable.timeDelta(userId, time);
//            }
//            return daysPredictionToExact(userId, time, daysPrediction);
        }

        private double expectedTime(final int userId, final double time) {
            double timeExpectation = timeApplicable.timeDelta(userId, time) *
                    daysApplicable.probabilityInterval(userId, 0, 1);
            for (int i = 1; i < Util.CHURN_THRESHOLD_DAYS; i++) {
                timeExpectation += daysPredictionToExact(userId, time, i) *
                        daysApplicable.probabilityInterval(userId, i, i + 1);
            }
            timeExpectation /= daysApplicable.probabilityBeforeX(userId, Util.CHURN_THRESHOLD_DAYS);
            return timeExpectation;
        }
    }

    @Override
    public ApplicableModel getApplicable() {
        return new ApplicableImpl(daysModel.getApplicable(), timeModel.getApplicable());
    }

    public static void calcDayPoints(final List<Event> events,
                                     final TIntIntMap userDayBorders,
                                     final TIntIntMap userDayPeaks) {
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

    public static TIntDoubleMap calcAvgStarts(final List<Event> events, final TIntIntMap userBorders) {
        final TIntDoubleMap lastDays = new TIntDoubleHashMap();
        final TIntDoubleMap starts = new TIntDoubleHashMap();
        final TIntIntMap count = new TIntIntHashMap();
        for (final Session session : DataPreprocessor.groupEventsToSessions(events)) {
            final int userId = session.userId();
            final double dayStart = Util.getDay(session.getStartTs(), userBorders.get(userId));
            final double dayTime = session.getStartTs() - dayStart;
            if (!lastDays.containsKey(userId) || lastDays.get(userId) != dayStart) {
                starts.adjustOrPutValue(userId, dayTime, dayTime);
                count.adjustOrPutValue(userId, 1, 1);
                lastDays.put(userId, dayStart);
            }
        }
        final TIntDoubleMap avgStarts = new TIntDoubleHashMap();
        for (final int userId : starts.keys()) {
            avgStarts.put(userId, starts.get(userId) / count.get(userId));
        }
        return avgStarts;
    }

    private static long combine(int user, int day) {
        return ((long) day) << 32 | user;
    }

    private static int getUser(long userDay) {
        return (int) userDay;
    }

    private static int getDay(long userDay) {
        return (int) (userDay >> 32);
    }

    private long sessionUserDays(Session session) {
        return combine(session.userId(), Util.getDaysFromPrevSession(session, userDayBorders.get(session.userId())));
    }

    private void calcDayAverages(List<Event> events) {
        userDayAverageDeltas = new TLongDoubleHashMap();
        DataPreprocessor.groupEventsToSessions(events).stream()
                .filter(Util::forPrediction)
                .collect(Collectors.groupingBy(this::sessionUserDays, Collectors.averagingDouble(Session::getDelta)))
                .forEach(userDayAverageDeltas::put);
    }

    @Override
    protected void write(final ObjectOutputStream objectOutputStream) throws IOException {
        writeBase(objectOutputStream);

        objectOutputStream.writeObject(userDayBorders);
        objectOutputStream.writeObject(userDayPeaks);
        objectOutputStream.writeObject(userDayAvgStarts);
        objectOutputStream.writeObject(userDayAverageDeltas);

        daysModel.write(objectOutputStream);
        timeModel.write(objectOutputStream);
    }

    public static ModelCombined load(final Path path) throws IOException, ClassNotFoundException {
        final ObjectInputStream objectInputStream = new ObjectInputStream(Files.newInputStream(path));
        final ModelCombined model = read(objectInputStream);
        objectInputStream.close();
        return model;
    }

    public static ModelCombined read(final ObjectInputStream objectInputStream) throws IOException, ClassNotFoundException {
        final int dim = objectInputStream.readInt();
        final double beta = objectInputStream.readDouble();
        final double eps = objectInputStream.readDouble();
        final double otherItemImportance = objectInputStream.readDouble();
        final DoubleUnaryOperator lambdaTransform = (DoubleUnaryOperator) objectInputStream.readObject();
        final DoubleUnaryOperator lambdaDerivativeTransform = (DoubleUnaryOperator) objectInputStream.readObject();
        final LambdaStrategyFactory lambdaStrategyFactory = (LambdaStrategyFactory) objectInputStream.readObject();

        final TIntIntMap userDayBorders = (TIntIntMap) objectInputStream.readObject();
        final TIntIntMap userDayPeaks = (TIntIntMap) objectInputStream.readObject();
        final TIntDoubleMap userDayAvgStarts = (TIntDoubleMap) objectInputStream.readObject();
        final TLongDoubleMap userDayAverageDeltas = (TLongDoubleMap) objectInputStream.readObject();

        final Model daysModel = ModelExpPerUser.read(objectInputStream);
        final Model timeModel = ConstantNextTimeModel.read(objectInputStream);

        final ModelCombined model = new ModelCombined(dim, beta, eps, otherItemImportance, lambdaTransform,
                lambdaDerivativeTransform, lambdaStrategyFactory, daysModel, timeModel, userDayBorders, userDayPeaks,
                userDayAvgStarts, userDayAverageDeltas);
        model.initModel();
        return model;
    }
}
