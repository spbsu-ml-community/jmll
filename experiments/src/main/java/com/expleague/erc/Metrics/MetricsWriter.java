package com.expleague.erc.Metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

public class MetricsWriter implements Model.FitListener {
    private static final String HIST_FILE_NAME = "history.txt";

    private final List<Event> trainData;
    private final List<Event> testData;
    private final double eps;
    private final Metric mae = new MAE();
    private final Metric spu = new SPU();
    private final Metric ll;
    private final Path histPath;

    public MetricsWriter(List<Event> trainData, List<Event> testData, double eps, Path saveDir) {
        this.trainData = trainData;
        this.testData = testData;
        this.eps = eps;
        ll = new LogLikelihood(eps);
        histPath = saveDir.resolve(HIST_FILE_NAME);
    }

    @Override
    public void apply(Model model) {
        final double[] maes = new double[2];
        final double[] lls = new double[2];
        final ForkJoinTask maeTask = ForkJoinPool.commonPool().submit(() -> {
            final Model.Applicable applicable = model.getApplicable();
            maes[0] = mae.calculate(trainData, applicable);
            maes[1] = mae.calculate(testData, applicable);
        });
        final ForkJoinTask llTask = ForkJoinPool.commonPool().submit(() -> {
            final Model.Applicable applicable = model.getApplicable();
            lls[0] = ll.calculate(trainData, applicable);
            lls[1] = ll.calculate(testData, applicable);
        });
        final ForkJoinTask<Double> spusTrainTask = ForkJoinPool.commonPool().submit(() ->
            spu.calculate(trainData, model.getApplicable()));
        final ForkJoinTask<Double> spusTestTask = ForkJoinPool.commonPool().submit(() ->
            spu.calculate(testData, model.getApplicable(trainData)));
        final ForkJoinTask histSaveTask = ForkJoinPool.commonPool().submit(() ->
                saveHist(model.getApplicable()));

        try {
            maeTask.join();
            llTask.join();
            final double spusTrain = spusTrainTask.get();
            final double spusTest = spusTestTask.get();
            System.out.printf("train_ll: %f, test_ll: %f, train_mae: %f, test_mae: %f, train_spu: %f, test_spu: %f\n",
                    lls[0], lls[1], maes[0], maes[1], spusTrain, spusTest);
            histSaveTask.join();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }

    private void saveHist(Model.Applicable applicable) {
        final StringBuilder histDescBuilder = new StringBuilder();
        for (final Event event : trainData) {
            final int userId = event.userId();
            final int itemId = event.itemId();
            final long pair = event.getPair();
            final double prDelta = event.getPrDelta();
            if (prDelta >= 0) {
                final double lambda = applicable.getLambda(userId, itemId);
                final double prediction = applicable.timeDelta(userId, itemId);
                final double probabilityLog = Math.log(-Math.exp(-lambda * (prDelta + eps)) +
                        Math.exp(-lambda * Math.max(0, prDelta - eps)));
                histDescBuilder.append(pair).append(" ").append(prDelta).append(" ").append(prediction)
                        .append(" ").append(lambda).append(" ").append(probabilityLog).append("\t");
            }
            applicable.accept(event);
        }
        histDescBuilder.append("\n");
        try {
            Files.write(histPath, histDescBuilder.toString().getBytes(), StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
