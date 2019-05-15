package com.expleague.erc.metrics;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.erc.Event;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.Model;
import gnu.trove.map.TIntObjectMap;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

public class MetricsWriter implements Model.FitListener {
    private static final String HIST_FILE_NAME = "history.txt";

    private final List<Event> trainData;
    private final List<Event> testData;
    private final Metric ll;
    private final Metric mae;
    private final Metric spu;
    private final boolean printEmbeddingsNorm;
    private final ForkJoinPool pool = ForkJoinPool.commonPool();

    public MetricsWriter(List<Event> trainData, List<Event> testData, Metric ll, Metric mae, Metric spu, boolean norm) {
        this.trainData = trainData;
        this.testData = testData;
        this.ll = ll;
        this.mae = mae;
        this.spu = spu;
        this.printEmbeddingsNorm = norm;
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
        ForkJoinTask maeTask = null, llTask = null;
        if (ll != null) {
            llTask = pool.submit(() -> {
                final ApplicableModel applicable = model.getApplicable();
                lls[0] = ll.calculate(trainData, applicable);
                lls[1] = ll.calculate(testData, applicable);
            });
        }
        if (mae != null) {
            maeTask = pool.submit(() -> {
                final ApplicableModel applicable = model.getApplicable();
                maes[0] = mae.calculate(trainData, applicable);
                maes[1] = mae.calculate(testData, applicable);
            });
        }
        ForkJoinTask<Double> spusTrainTask = null, spusTestTask = null;
        if (spu != null) {
            spusTrainTask = pool.submit(() -> spu.calculate(trainData, model.getApplicable()));
            spusTestTask = pool.submit(() -> spu.calculate(testData, model.getApplicable(trainData)));
        }

        try {
            if (ll != null) {
                llTask.join();
                System.out.printf("train_ll: %f, test_ll: %f, ", lls[0], lls[1]);
            }
            if (mae != null) {
                maeTask.join();
                System.out.printf("train_mae: %f, test_mae: %f, ", maes[0], maes[1]);
            }
            if (spu != null) {
                System.out.printf("train_spu: %f, test_spu: %f, ", spusTrainTask.get(), spusTestTask.get());
            }
            if (printEmbeddingsNorm) {
                System.out.printf("user_norm: %f, item_norm: %f",
                        meanEmbeddingNorm(model.getUserEmbeddings()), meanEmbeddingNorm(model.getItemEmbeddings()));
            }
            System.out.println();
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }
    }
}
