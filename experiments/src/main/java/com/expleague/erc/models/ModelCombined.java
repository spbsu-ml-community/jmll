package com.expleague.erc.models;

import com.expleague.erc.*;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategyFactory;
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
import java.util.function.DoubleUnaryOperator;

import static com.expleague.erc.Util.DAY_HOURS;

public class ModelCombined extends Model {
    private final TIntIntMap userDayBorders;
    private final TIntIntMap userDayPeaks;
    private TIntDoubleMap userDayAvgStarts;

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
                          TIntIntMap userDayBorders, TIntIntMap userDayPeaks, TIntDoubleMap userDayAvgStarts) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
        this.daysModel = daysModel;
        this.timeModel = timeModel;
        this.userDayBorders = userDayBorders;
        this.userDayPeaks = userDayPeaks;
        this.userDayAvgStarts = userDayAvgStarts;
    }

    @Override
    public void initModel(final List<Event> events) {
        if (isInit) {
            return;
        }
        daysModel.makeInitialEmbeddings(events);
        timeModel.makeInitialEmbeddings(events);
        initIds();
        userDayAvgStarts = calcAvgStarts(events);
        calcDayPoints(events, userDayBorders, userDayPeaks);
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
//        daysModel.fit(events, learningRate, iterationsNumber, decay, listener);
//        timeModel.fit(events, learningRate, iterationsNumber, decay, listener);
        initModel(events);
        daysModel.initModel(events);
        timeModel.initModel(events);
        double lr = learningRate;
        if (fitListener != null) {
            fitListener.apply(this);
        }
        for (int i = 0; i < iterationsNumber; i++) {
            daysModel.fit(events, lr, 1, decay);
            timeModel.fit(events, lr, 1, decay);
            lr *= decay;
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

        @Override
        public double timeDelta(final int userId, final double time) {
            final double dayPrediction = daysApplicable.timeDelta(userId, time);
            if (time + dayPrediction * DAY_HOURS < Util.getDay(time, userDayBorders.get(userId)) + DAY_HOURS) {
                return timeApplicable.timeDelta(userId, time);
            }
            final int predictedDay = (int) (time + (int) dayPrediction * DAY_HOURS) / DAY_HOURS;
            final double predictedTime = predictedDay * DAY_HOURS + userDayAvgStarts.get(userId);
            return predictedTime - time;
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


    @Override
    protected void write(final ObjectOutputStream objectOutputStream) throws IOException {
        objectOutputStream.writeInt(dim);
        objectOutputStream.writeDouble(beta);
        objectOutputStream.writeDouble(eps);
        objectOutputStream.writeDouble(otherItemImportance);
        objectOutputStream.writeObject(lambdaTransform);
        objectOutputStream.writeObject(lambdaDerivativeTransform);
        objectOutputStream.writeObject(lambdaStrategyFactory);

        objectOutputStream.writeObject(userDayBorders);
        objectOutputStream.writeObject(userDayPeaks);
        objectOutputStream.writeObject(userDayAvgStarts);

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

        final Model daysModel = ModelExpPerUser.read(objectInputStream);
        final Model timeModel = ConstantNextTimeModel.read(objectInputStream);

        final ModelCombined model =
                new ModelCombined(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform,
                        lambdaStrategyFactory, daysModel, timeModel, userDayBorders, userDayPeaks, userDayAvgStarts);
        model.initModel();
        return model;
    }
}
