package com.expleague.ml.methods.nn;

import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.IntSeq;
import com.expleague.ml.BFGrid;
import com.expleague.ml.GridTools;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.data.set.VecDataSet;
import com.expleague.ml.data.set.impl.VecDataSetImpl;
import com.expleague.ml.factorization.impl.StochasticALS;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.func.ScaledVectorFunc;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.WeightedLoss;
import com.expleague.ml.loss.blockwise.BlockwiseMLLLogit;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.Optimization;
import com.expleague.ml.methods.greedyRegion.GreedyProbLinearRegion;
import com.expleague.ml.methods.greedyRegion.GreedyProbLinearRegion.ProbRegion;
import com.expleague.ml.methods.multiclass.gradfac.GradFacMulticlass;
import com.expleague.ml.models.nn.ConvNet;

import java.io.PrintStream;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.IntStream;

public class NeuralTreesOptimization implements Optimization<BlockwiseMLLLogit, VecDataSet, Vec> {
  private int numIterations;
  private final ConvNet nn;
  private final FastRandom rng;
  private int sgdIterations;
  private int numTrees;
  private final int batchSize;
  private final WeightDumper weightDumper;
  private final PrintStream debug;
  private double sgdStep;
  private double boostingStep;
  private VecDataSet test;
  private BlockwiseMLLLogit testLoss;

  public NeuralTreesOptimization(int numIterations, int sgdIterations, int batchSize,
                                 double sgdStep, int numTrees, double boostingStep,
                                 ConvNet nn, FastRandom rng, WeightDumper weightDumper, PrintStream debug) {
    this.numIterations = numIterations;
    this.sgdIterations = sgdIterations;
    this.batchSize = batchSize;
    this.numTrees = numTrees;
    this.boostingStep = boostingStep;
    this.nn = nn;
    this.rng = rng;
    this.weightDumper = weightDumper;
    this.debug = debug;
    this.sgdStep = sgdStep;

    debug.printf("parameters:\n" +
        "    numIterations = %d;\n" +
        "    sgdIterations = %d;\n" +
        "    batchSize = %d;\n" +
        "    numTrees = %d;\n" +
        "    boostingStep = %f;\n" +
        "    sgdStep = %f;\n", numIterations, sgdIterations,
        batchSize, numTrees, boostingStep, sgdStep);
  }

  public void setTest(VecDataSet test, IntSeq labels) {
    this.test = test;
    testLoss = new BlockwiseMLLLogit(labels, test);
  }

  @Override
  public Function<Vec, Vec> fit(VecDataSet learn, BlockwiseMLLLogit loss) {
    final HighLevelDataset allLearn = HighLevelDataset.createFromDs(learn, loss, nn);
    final DataNormalizer normalizer = allLearn.normalizer;
    final HighLevelDataset allTest = HighLevelDataset.createFromDs(test, normalizer, testLoss, nn);

    for (int iter = 0; iter < numIterations; iter++) {
      if ((iter + 1) % 10 == 0) {
        weightDumper.dump(iter);
      }
//      final HighLevelDataset highLearn = HighLevelDataset.sampleFromDataset(learn, normalizer, loss, nn, nSampleBuildTree, rng);
//      final HighLevelDataset highTest = HighLevelDataset.sampleFromDataset(test, normalizer, testLoss, nn, nSampleBuildTree, rng);
      final Ensemble ensemble = fitBoosting(allLearn, allTest);

      {
        Vec L = new ArrayVec(nn.wdim());
        Vec grad = new ArrayVec(nn.wdim());
        Vec step = new ArrayVec(nn.wdim());
        VecTools.fill(L, this.sgdStep);

        final Vec prevGrad = new ArrayVec(nn.wdim());
        final Vec partial = new ArrayVec(nn.wdim());

        for (int sgdIter = 0; sgdIter < sgdIterations; sgdIter++) {
          VecTools.fill(grad, 0);

          double batchLoss = 0.;
          for (int i = 0; i < batchSize; i++) {
            final int sampleIdx = rng.nextInt(learn.length());
            Vec apply = nn.apply(learn.data().row(sampleIdx));
            normalizer.transTo(apply, apply);

            batchLoss += loss.value(ensemble.trans(apply), sampleIdx);

            final Vec treeGrad = ensembleGradient(ensemble, loss, apply, sampleIdx);
            final Vec baseVec = learn.data().row(sampleIdx);
            nn.gradientTo(baseVec, new TargetByTreeOut(treeGrad), partial);

            VecTools.append(grad, partial);
          }

          VecTools.assign(step, grad);
          VecTools.scale(step, L);
          VecTools.incscale(nn.weights(), step, sgdStep);
          for (int i = 0; i < L.dim(); i++) {
            L.set(i, Math.min(L.get(i) / 0.99, 1 / Math.abs(grad.get(i))));
          }

          if (sgdIter % 10 == 0) {
            System.out.println("sgd [" + sgdIter + "] ll on batch: " + Math.exp(batchLoss / batchSize));
            debug.println("Grad alignment: " + VecTools.cosine(prevGrad, grad));
            VecTools.assign(prevGrad, grad);
          }

        }
//        if ((sgdIter + 1) % 100 == 0) {
        {
          allLearn.update();
          allTest.update();

          final Mx resultTrain = ensemble.transAll(allLearn.data());
          final Mx resultTest = ensemble.transAll(allTest.data());
          final double lTrain = allLearn.loss().value(resultTrain);
          final double lTest = allTest.loss().value(resultTest);
          final double accTest = accuracy(allTest.loss(), resultTest);
          debug.println("after sgd: loss(train): " + lTrain +
              " loss(test): " + lTest + " acc(test): " + accTest);

        }
      }
    }

    final Vec xOpt = nn.weights();
    boostingStep /= 2.;
    numTrees = 100000;
    final Ensemble ensemble = fitBoosting(allLearn, allTest);

    return argument -> {
      Vec result = nn.apply(argument, xOpt);
      return ensemble.trans(result);
    };
  }

  private Vec ensembleGradient(Ensemble ensemble, BlockwiseMLLLogit loss, Vec x, int blockId) {
    double epsilon = 1e-10;

    final Vec ensembleGrad = new ArrayVec(nn.ydim());

    final Vec lossGrad = new ArrayVec(loss.blockSize());
    final Vec treeOut = ensemble.trans(x);
    loss.gradient(treeOut, lossGrad, blockId);

    final Vec currentWeights = new ArrayVec(loss.blockSize());
    final Vec grad = new ArrayVec(nn.ydim());
    for (int i = 0; i < ensemble.models.length; i++) {
      VecTools.fill(grad, 0.);
      final ScaledVectorFunc model = (ScaledVectorFunc) ensemble.models[i];
      VecTools.assign(currentWeights, model.weights);
      VecTools.scale(currentWeights, lossGrad);
      ((ProbRegion) model.function).gradientTo(x, grad);
//      {
        //      final Vec realGrad = new ArrayVec(nn.ydim());
        //      double value = model.function.value(x);
        //      for (int j = 0; j < grad.dim(); j++) {
        //        x.adjust(j, epsilon);
        //        double valuePrime = model.function.value(x);
        //        realGrad.set(j, (valuePrime - value) / epsilon);
        //        x.adjust(j, -epsilon);
        //      }
//      }
      VecTools.scale(grad, ensemble.weights.get(i) * VecTools.sum(currentWeights));
      VecTools.append(ensembleGrad, grad);
    }

    return ensembleGrad;
  }

  private Ensemble fitBoosting(HighLevelDataset learn, HighLevelDataset test) {
    final BFGrid grid = GridTools.medianGrid(learn.vec(), 32);
    final GreedyProbLinearRegion<WeightedLoss<L2Reg>> weak = new GreedyProbLinearRegion<>(grid, 7);
    final BootstrapOptimization bootstrap = new BootstrapOptimization(weak, rng);

    final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(new GradFacMulticlass(
        bootstrap, new StochasticALS(rng, 1000.), L2Reg.class, false), L2Reg.class, numTrees, boostingStep);

    final Consumer<Trans> counter = new ProgressHandler() {
      int index = 0;
      Mx currentLearn = new VecBasedMx(learn.data().rows(), learn.loss.blockSize());
      Mx currentTest = new VecBasedMx(test.data().rows(), learn.loss.blockSize());

      @Override
      public void accept(Trans partial) {
        index++;
        Mx resultTrain = null;
        Mx resultTest = null;
        if (partial instanceof Ensemble) {
          Ensemble ensemble = (Ensemble) partial;
          Trans last = ensemble.last();
          VecTools.incscale(currentLearn, last.transAll(learn.data()), ensemble.wlast());
          resultTrain = currentLearn;
          VecTools.incscale(currentTest, last.transAll(test.data()), ensemble.wlast());
          resultTest = currentTest;
        }

        if (index % 100 == 0) {
          if (resultTest == null || resultTrain == null){
            resultTrain = partial.transAll(learn.data());
            resultTest = partial.transAll(test.data());
          }

          final double lTrain = learn.loss().value(resultTrain);
          final double lTest = test.loss().value(resultTest);
          final double accTest = accuracy(test.loss(), resultTest);
          debug.println("boost [" + (index) + "], ll(train): " + lTrain +
              " ll(test): " + lTest + " acc(test): " + accTest);
        }
      }
    };
    boosting.addListener(counter);

    final Ensemble ensemble = boosting.fit(learn.vec(), learn.loss());

    final Vec result = ensemble.transAll(learn.data()).vec();
    final double curLossValue = learn.loss().value(result);
    debug.println("ensemble ll: " + curLossValue);

    return ensemble;
  }

  private static double accuracy(BlockwiseMLLLogit loss, Mx results) {
    final Vec predict = new ArrayVec(results.rows());
    IntStream.range(0, results.rows()).parallel().forEach(i -> {
      final Vec prob = loss.prob(results.row(i), new ArrayVec(loss.blockSize() + 1));
      predict.set(i, VecTools.argmax(prob));
    });
    final IntSeq labels = loss.labels();
    int acc = 0;
    for (int i = 0; i < predict.dim(); i++) {
      acc += predict.get(i) == labels.intAt(i) ? 1 : 0;
    }
    return ((double) acc) / results.rows();
  }

  public static class WeightDumper {
    private final String path;
    private final ConvNet nn;

    public WeightDumper(ConvNet nn, String path) {
      this.nn = nn;
      this.path = path;
    }

    public void dump(int iter) {
      nn.save(path + "_iter_" + iter + ".nn");
    }
  }

  private static class HighLevelDataset {
    private final ConvNet nn;
    private final VecDataSet base;
    private final BlockwiseMLLLogit loss;
    private final int[] sampleIdxs;
    private DataNormalizer normalizer;
    private Mx highData;


    private HighLevelDataset(Mx highData, DataNormalizer normalizer, ConvNet nn, VecDataSet base, BlockwiseMLLLogit loss, int[] sampleIdxs) {
      this.highData = highData;
      this.normalizer = normalizer;
      this.nn = nn;
      this.base = base;
      this.loss = loss;
      this.sampleIdxs = sampleIdxs;
    }

    static HighLevelDataset sampleFromDataset(VecDataSet ds, BlockwiseMLLLogit loss, ConvNet nn, int numSamples, FastRandom rng) {
      return sampleFromDataset(ds, null, loss, nn, numSamples, rng);
    }

    static HighLevelDataset sampleFromDataset(VecDataSet ds, DataNormalizer normalizer, BlockwiseMLLLogit loss, ConvNet nn, int numSamples, FastRandom rng) {
      Mx highData = new VecBasedMx(numSamples, nn.ydim());
      final int[] sampleIdx = new int[numSamples];

      for (int i = 0; i < numSamples; i++) {
        sampleIdx[i] = rng.nextInt(ds.length());
        final Vec result = nn.apply(ds.data().row(sampleIdx[i]));
        VecTools.assign(highData.row(i), result);
      }

      if (normalizer == null)
        normalizer = new DataNormalizer(highData, 0, 3.);
      highData = normalizer.transAll(highData, true);

      final Vec target = new ArrayVec(numSamples);
      IntStream.range(0, numSamples).forEach(idx -> target.set(idx, loss.label(sampleIdx[idx])));
      final BlockwiseMLLLogit newLoss = new BlockwiseMLLLogit(target, ds);

      return new HighLevelDataset(highData, normalizer, nn, ds, newLoss, sampleIdx);
    }

    static HighLevelDataset createFromDs(VecDataSet ds, DataNormalizer normalizer, BlockwiseMLLLogit loss, ConvNet nn) {
      Mx highData = new VecBasedMx(ds.length(), nn.ydim());
      for (int i = 0; i < ds.length(); i++) {
        final Vec result = nn.apply(ds.data().row(i));
        VecTools.assign(highData.row(i), result);
      }

      if (normalizer == null) {
        normalizer = new DataNormalizer(highData, 0., 3.);
      }
      highData = normalizer.transAll(highData, true);

      return new HighLevelDataset(highData, normalizer, nn, ds, loss, IntStream.range(0, ds.length()).toArray());
    }

    static HighLevelDataset createFromDs(VecDataSet ds, BlockwiseMLLLogit loss, ConvNet nn) {
      return createFromDs(ds, null, loss, nn);
    }

    public Mx data() {
      return highData;
    }

    public void setNormalizer(DataNormalizer normalizer) {
      this.normalizer = normalizer;
    }

    public DataNormalizer getNormalizer() {
      return normalizer;
    }

    public BlockwiseMLLLogit loss() {
      return loss;
    }

    public VecDataSet vec() {
      return new VecDataSetImpl(highData, base);
    }

    public Vec baseVecById(int id) {
      return base.data().row(sampleIdxs[id]);
    }

    public void update() {
      for (int i = 0; i < highData.rows(); i++) {
        final Vec result = nn.apply(base.data().row(sampleIdxs[i]));
        VecTools.assign(highData.row(i), result);
      }
      highData = normalizer.transAll(highData, true);
    }

    public Vec getCached(int sampleIdx) {
      return highData.row(sampleIdx);
    }

    public Vec get(int idx) {
      final Vec apply = nn.apply(base.data().row(sampleIdxs[idx]));
      normalizer.transTo(apply, apply);
      return apply;
    }
  }

  private static class DataNormalizer extends Trans.Stub {
    private final Vec mean;
    private final Vec disp;
    private final double newMean;
    private final double newDisp;

    DataNormalizer(Mx data, double newMean, double newDisp) {
      this.newMean = newMean;
      this.newDisp = newDisp;
      final int featuresDim = data.columns();

      mean = VecTools.fill(new ArrayVec(featuresDim), 0.);
      disp = VecTools.fill(new ArrayVec(featuresDim), 0.);

      for (int i = 0; i < data.rows(); i++) {
        final Vec row = data.row(i);
        VecTools.append(mean, row);
        appendSqr(disp, row, 1.);
      }

      VecTools.scale(mean, 1. / data.rows());
      VecTools.scale(disp, 1. / data.rows());
      appendSqr(disp, mean, -1.);

      for (int i = 0; i < featuresDim; i++) {
        double v = disp.get(i) == 0. ? 1. : disp.get(i);
        disp.set(i, v);
      }
    }

    @Override
    public Vec transTo(Vec x, Vec to) {
      if (x.dim() != mean.dim()) {
        throw new IllegalArgumentException();
      }

      for (int i = 0; i < x.dim(); i++) {
        final double v = (x.get(i) - mean.get(i)) / Math.sqrt(disp.get(i)) * newDisp + newMean;
        to.set(i, v);
      }

      return to;
    }

    private void appendSqr(Vec to, Vec who, double alpha) {
      for (int i = 0; i < to.dim(); i++) {
        final double v = who.get(i);
        to.adjust(i, alpha * v * v);
      }
    }

    @Override
    public int xdim() {
      return mean.dim();
    }

    @Override
    public int ydim() {
      return xdim();
    }
  }

  private class TargetByTreeOut extends FuncC1.Stub {
    private final Vec treesGradient;

    TargetByTreeOut(Vec gradient) {
      this.treesGradient = gradient;
    }

    @Override
    public double value(Vec x) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Vec gradientTo(Vec x, Vec to) {
      VecTools.assign(to, treesGradient);
      return to;
    }

    @Override
    public int dim() {
      return treesGradient.dim();
    }
  }
}
