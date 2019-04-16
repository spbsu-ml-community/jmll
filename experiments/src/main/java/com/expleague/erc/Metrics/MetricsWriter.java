package com.expleague.erc.Metrics;

import com.expleague.erc.Event;
import com.expleague.erc.Model;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

public class MetricsWriter implements Model.FitListener {
    private final List<Event> trainData;
    private final List<Event> testData;

    private final Metric mae = new MAE();
    private final Metric spu = new SPU();
    private final Metric ll;

    public MetricsWriter(List<Event> trainData, List<Event> testData, double eps) {
        this.trainData = trainData;
        this.testData = testData;
        ll = new LogLikelihood(eps);
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


        try {
            maeTask.join();
            llTask.join();
            final double spusTrain = spusTrainTask.get();
            final double spusTest = spusTestTask.get();
            System.out.printf("train_ll: %f, test_ll: %f, train_mae: %f, test_mae: %f, train_spu: %f, test_spu: %f\n",
                    lls[0], lls[1], maes[0], maes[1], spusTrain, spusTest);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }
}
