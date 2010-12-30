package ml.methods;

import ml.Model;
import ml.data.DataSet;
import ml.loss.LossFunction;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * User: solar
 * Date: 23.12.2010
 * Time: 16:24:43
 */
public abstract class ParallelByFeatureMethod implements MLMethod {
    public interface FeatureFilter {
        boolean relevant(int findex);
    }

    protected abstract Model fit(DataSet learn, LossFunction loss, FeatureFilter filter);
    private final int cores = Runtime.getRuntime().availableProcessors();
    ThreadPoolExecutor executor = new ThreadPoolExecutor(cores, cores, 1, TimeUnit.DAYS, new ArrayBlockingQueue<Runnable>(cores));
    public Model fit(final DataSet learn, final LossFunction loss) {
        final int threadsCount = Runtime.getRuntime().availableProcessors();
        final Model[] bestModels = new Model[threadsCount];
        final CountDownLatch latch = new CountDownLatch(threadsCount);
        for (int i = 0; i < threadsCount; i++) {
            final int fi = i;
            executor.execute(new Runnable() {
                public void run() {
                    bestModels[fi] = fit(learn, loss, new FeatureFilter() {
                        public boolean relevant(int featureIndex) {
                            return featureIndex % threadsCount == fi;
                        }
                    });
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        }
        catch (InterruptedException ignored) {}
        Model result = bestModels[0];
        for (int i = 0; i < threadsCount; i++) {
            if (result == null || (bestModels[i] != null && result.learnScore() > bestModels[i].learnScore())) {
                result = bestModels[i];
            }
        }
        return result;
    }
}
