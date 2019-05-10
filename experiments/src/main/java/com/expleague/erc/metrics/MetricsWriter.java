package com.expleague.erc.metrics;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.erc.Event;
import com.expleague.erc.EventSeq;
import com.expleague.erc.Session;
import com.expleague.erc.Util;
import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.Model;
import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

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
    private final Metric mae = new MAEPerUser();
    private final Metric spu = new SPUPerUser();
    private final Metric ll;
    private final Path histPath;

    public MetricsWriter(List<Event> trainData, List<Event> testData, double eps, Path saveDir) {
        this.trainData = trainData;
        this.testData = testData;
        this.eps = eps;
        ll = new LogLikelihoodDaily(trainData);
        histPath = saveDir.resolve(HIST_FILE_NAME);
    }

    private double meanEmbeddingNorm(final TIntObjectMap<Vec> embeddings) {
        return embeddings.valueCollection().stream()
                .mapToDouble(VecTools::norm)
                .average().orElse(-1);
    }

    @Override
    public void apply(Model model) {
        final double[] maes = new double[2];
        final double[] lls = new double[2];
        final ForkJoinTask maeTask = ForkJoinPool.commonPool().submit(() -> {
            final ApplicableModel applicable = model.getApplicable();
            maes[0] = mae.calculate(trainData, applicable);
            maes[1] = mae.calculate(testData, applicable);
        });
        final ForkJoinTask llTask = ForkJoinPool.commonPool().submit(() -> {
            final ApplicableModel applicable = model.getApplicable();
            lls[0] = ll.calculate(trainData, applicable);
            lls[1] = ll.calculate(testData, applicable);
        });
        final ForkJoinTask<Double> spusTrainTask = ForkJoinPool.commonPool().submit(() ->
            spu.calculate(trainData, model.getApplicable()));
        final ForkJoinTask<Double> spusTestTask = ForkJoinPool.commonPool().submit(() ->
            spu.calculate(testData, model.getApplicable(trainData)));
//        final ForkJoinTask histSaveTask = ForkJoinPool.commonPool().submit(() ->
//                saveHist(model.getApplicable()));

        try {
            maeTask.join();
            llTask.join();
            final double spusTrain = spusTrainTask.get();
            final double spusTest = spusTestTask.get();
            System.out.printf("train_ll: %f, test_ll: %f, train_mae: %f, test_mae: %f, " +
                            "train_spu: %f, test_spu: %f, " +
                            "user_norm: %f, item_norm, %f\n",
                    lls[0], lls[1], maes[0], maes[1], spusTrain, spusTest,
                    meanEmbeddingNorm(model.getUserEmbeddings()),
                    meanEmbeddingNorm(model.getItemEmbeddings()));
//            histSaveTask.join();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }

    private void saveHist(ApplicableModel applicable) {
        final StringBuilder histDescBuilder = new StringBuilder();
        for (final EventSeq eventSeq : DataPreprocessor.groupToEventSeqs(trainData)) {
            final int userId = eventSeq.userId();
            final int itemId = eventSeq.itemId();
            double prDelta = eventSeq.getDelta();
            if (prDelta >= 0) {
                final double lambda = applicable.getLambda(userId, itemId);
                final double prediction = applicable.timeDelta(userId, itemId);
                prDelta = Math.max(prDelta, eps);
                final double pLog =
                        Math.log(applicable.probabilityInterval(userId, itemId, prDelta - eps, prDelta + eps));
                histDescBuilder.append(userId).append(" ").append(itemId).append(" ").append(prDelta).append(" ")
                        .append(prediction).append(" ").append(lambda).append(" ").append(pLog).append("\t");
            }
            applicable.accept(eventSeq);
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
