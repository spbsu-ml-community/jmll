package com.expleague.erc.models;

import com.expleague.erc.*;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.lambda.LambdaStrategyFactory;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

import static com.expleague.erc.Util.DAY_HOURS;

public class ConstantNextTimeModel extends Model {
    private TIntDoubleMap averageOneDayDelta;

    public ConstantNextTimeModel(int dim, double beta, double eps, double otherItemImportance,
                                 DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                                 LambdaStrategyFactory lambdaStrategyFactory) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
    }

    private ConstantNextTimeModel(int dim, double beta, double eps, double otherItemImportance,
                                  DoubleUnaryOperator lambdaTransform, DoubleUnaryOperator lambdaDerivativeTransform,
                                  LambdaStrategyFactory lambdaStrategyFactory, TIntDoubleMap averageOneDayDelta) {
        super(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform, lambdaStrategyFactory);
        this.averageOneDayDelta = averageOneDayDelta;
    }

    @Override
    public void initModel(final List<Event> events) {
        if (!isInit) {
            averageOneDayDelta = calcAverageOneDayDelta(events);
            isInit = true;
        }
    }

    private class ApplicableImpl implements ApplicableModel {
        private ApplicableImpl() {}

        @Override
        public void accept(final EventSeq eventSeq) {}

        @Override
        public double timeDelta(final int userId, final double time) {
            return averageOneDayDelta.get(userId);
        }
    }

    @Override
    public void fit(final List<Event> events, final double learningRate, final int iterationsNumber,
                    final double decay, final FitListener listener) {}

    @Override
    public ApplicableModel getApplicable() {
        return new ApplicableImpl();
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

    protected void write(final ObjectOutputStream objectOutputStream) throws IOException {
        objectOutputStream.writeInt(dim);
        objectOutputStream.writeDouble(beta);
        objectOutputStream.writeDouble(eps);
        objectOutputStream.writeDouble(otherItemImportance);
        objectOutputStream.writeObject(lambdaTransform);
        objectOutputStream.writeObject(lambdaDerivativeTransform);
        objectOutputStream.writeObject(lambdaStrategyFactory);
        objectOutputStream.writeObject(averageOneDayDelta);
    }

    public static ConstantNextTimeModel load(final Path path) throws IOException, ClassNotFoundException {
        final ObjectInputStream objectInputStream = new ObjectInputStream(Files.newInputStream(path));
        final ConstantNextTimeModel model = read(objectInputStream);
        objectInputStream.close();
        return model;
    }

    public static ConstantNextTimeModel read(final ObjectInputStream objectInputStream) throws IOException, ClassNotFoundException {
        final int dim = objectInputStream.readInt();
        final double beta = objectInputStream.readDouble();
        final double eps = objectInputStream.readDouble();
        final double otherItemImportance = objectInputStream.readDouble();
        final DoubleUnaryOperator lambdaTransform = (DoubleUnaryOperator) objectInputStream.readObject();
        final DoubleUnaryOperator lambdaDerivativeTransform = (DoubleUnaryOperator) objectInputStream.readObject();
        final LambdaStrategyFactory lambdaStrategyFactory = (LambdaStrategyFactory) objectInputStream.readObject();
        final TIntDoubleMap averageOneDayDelta = (TIntDoubleMap) objectInputStream.readObject();
        final ConstantNextTimeModel model = new ConstantNextTimeModel(dim, beta, eps, otherItemImportance, lambdaTransform, lambdaDerivativeTransform,
                lambdaStrategyFactory, averageOneDayDelta);
        model.initModel();
        return model;
    }
}
