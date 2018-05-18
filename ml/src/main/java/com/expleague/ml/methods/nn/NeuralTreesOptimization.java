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
import com.expleague.ml.factorization.impl.ALS;
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

import java.util.Arrays;
import java.util.function.Consumer;
import java.util.function.Function;

public class NeuralTreesOptimization implements Optimization<BlockwiseMLLLogit, VecDataSet, Vec> {
  private final int numIterations;
  private final int nSampleBuildTree;
  private final ConvNet nn;
  private final FastRandom rng;
  private final int[] sampleIdxsTrain;
  private final int[] sampleIdxsTest;
  private BlockwiseMLLLogit loss;
  private Vec xOpt;
  private Ensemble ensemble;
  private int sgdIterations;

  public NeuralTreesOptimization(int numIterations, int nSampleBuildTree, int sgdIterations, ConvNet nn, FastRandom rng) {
    this.numIterations = numIterations;
    this.nSampleBuildTree = 1000;//nSampleBuildTree;
    this.sgdIterations = 10;//10;
    this.nn = nn;
    this.rng = rng;
    sampleIdxsTrain = new int[this.nSampleBuildTree];
    sampleIdxsTest = new int[this.nSampleBuildTree];
  }

  @Override
  public Function<Vec, Vec> fit(VecDataSet learn, BlockwiseMLLLogit loss) {
    this.loss = loss;
    final Mx highFeaturesTest = new VecBasedMx(nSampleBuildTree, nn.ydim());
    final Mx highFeaturesTrain = new VecBasedMx(nSampleBuildTree, nn.ydim());
    Vec prevGrad = new ArrayVec(nn.wdim());

    for (int i = 0; i < nSampleBuildTree; i++) {
      sampleIdxsTest[i] = rng.nextInt(learn.length());
    }

    for (int iter = 0; iter < numIterations; iter++) {
      for (int i = 0; i < nSampleBuildTree; i++) {
        final Vec result = nn.apply(learn.data().row(sampleIdxsTest[i]));
        VecTools.assign(highFeaturesTest.row(i), result);
      }

      for (int i = 0; i < nSampleBuildTree; i++) {
        sampleIdxsTrain[i] = rng.nextInt(learn.length());
        final Vec result = nn.apply(learn.data().row(sampleIdxsTrain[i]));
        VecTools.assign(highFeaturesTrain.row(i), result);
      }

      final VecDataSetImpl highLearn = new VecDataSetImpl(highFeaturesTrain, null);
      final BlockwiseMLLLogit curLossTrain = new BlockwiseMLLLogit(new IntSeq(Arrays.stream(sampleIdxsTrain).map(loss::label).toArray()), learn);
      final BlockwiseMLLLogit curLossTest = new BlockwiseMLLLogit(new IntSeq(Arrays.stream(sampleIdxsTest).map(loss::label).toArray()), learn);

      if (ensemble != null) {
        final Vec result = ensemble.transAll(highLearn.data()).vec();
        final double value = curLossTrain.value(result);
        System.out.println("[" + iter + "], lossAfterGrad: " + value);
      }

      final BFGrid grid = GridTools.medianGrid(highLearn, 64);
      final GreedyProbLinearRegion<WeightedLoss<L2Reg>> weak = new GreedyProbLinearRegion<>(grid, 6);
      final BootstrapOptimization bootstrap = new BootstrapOptimization(weak, rng);

      final GradientBoosting<BlockwiseMLLLogit> boosting = new GradientBoosting<>(new GradFacMulticlass(
          bootstrap, new ALS(150, 0.), L2Reg.class, false), L2Reg.class, 500, 0.1);

      final Consumer<Trans> counter = new ProgressHandler() {
        int index = 0;

        @Override
        public void accept(Trans partial) {
          index++;

          if (index % 100 == 0) {
            final Mx resultTest = partial.transAll(highFeaturesTest);
            final double lTest = curLossTest.value(resultTest);
            final Mx resultTrain = partial.transAll(highFeaturesTrain);
            final double lTrain = curLossTrain.value(resultTrain);
            System.out.println("boost [" + (index) + "], loss(train): " + lTrain + " loss(test): " + lTest);
          }
        }
      };
      boosting.addListener(counter);

      ensemble = boosting.fit(highLearn, curLossTrain);
//      System.out.println(((ScaledVectorFunc) ensemble.models[0]).function);

      final Vec gradTree = new ArrayVec(loss.blockSize());
      final Trans gradient = new Trans.Stub() {
        @Override
        public Vec transTo(Vec x, Vec to) {
          final Vec currentWeights = new ArrayVec(gradTree.dim());
          final Vec grad = new ArrayVec(nn.ydim());
          for (int i = 0; i < ensemble.models.length; i++) {
            final ScaledVectorFunc model = (ScaledVectorFunc) ensemble.models[i];
            VecTools.assign(currentWeights, model.weights);
            VecTools.scale(currentWeights, gradTree);
            ((ProbRegion) model.function).gradientTo(x, grad);
            VecTools.scale(grad, ensemble.weights.get(i) * VecTools.sum(currentWeights));
            VecTools.append(to, grad);
          }

          return to;
        }

        @Override
        public int xdim() {
          return 0;
        }

        @Override
        public int ydim() {
          return 0;
        }
      };

      final Vec result = ensemble.transAll(highLearn.data()).vec();
      final double curLossValue = curLossTrain.value(result);
      System.out.println("[" + iter + "], loss: " + curLossValue);
      double sgdStep = 1e-4;

      {
        final Vec grad = new ArrayVec(nn.wdim());
        final Vec x = nn.weights();
        for (int sgdIter = 0; sgdIter < 100000; sgdIter++) {

          VecTools.fill(grad, 0);
          Vec partial = new ArrayVec(nn.wdim());
          for (int i = 0; i < 128; i++) {
            int idx = sampleIdxsTrain[rng.nextInt(nSampleBuildTree)];
            final Vec nnResult = nn.apply(learn.data().row(idx));
            final Vec treeOut = ensemble.trans(nnResult);
            loss.gradient(treeOut, gradTree, idx);
            final double value = loss.value(ensemble.trans(nnResult), idx);
            final Vec treeGrad = new ArrayVec(nnResult.dim());
            gradient.transTo(nnResult, treeGrad);
//            final Vec realGrad = new ArrayVec(nnResult.dim());
//            for (int j = 0; j < nnResult.dim(); j++) {
//              final double epsilon = 1e-7;
//              nnResult.adjust(j, epsilon);
//              double newVal = loss.value(ensemble.trans(nnResult), idx);
//              nnResult.adjust(j, -epsilon);
//              realGrad.set(j, (newVal - value) / epsilon);
//            }
            nn.gradientTo(learn.data().row(idx), x, new TargetByTreeOut(treeGrad), partial);
            VecTools.append(grad, partial);
          }

          final int lastLayerWStart = nn.wdim() - 500 * 80;
          for (int i = 0; i < lastLayerWStart; i++) {
            grad.set(i, 0);
          }
          VecTools.scale(grad, sgdStep);
          VecTools.append(nn.weights(), grad);

            for (int i = 0; i < nSampleBuildTree; i++) {
              VecTools.assign(highFeaturesTrain.row(i), nn.apply(learn.data().row(sampleIdxsTrain[i])));
            }
            final Mx resultTrain = ensemble.transAll(highFeaturesTrain);
            final double lTrain = curLossTrain.value(resultTrain);
            System.out.println("sgd [" + (sgdIter) + "], loss(train): " + lTrain);
//          }
        }
        System.out.println("Grad alignment: " + VecTools.cosine(prevGrad, grad));
        VecTools.assign(prevGrad, grad);
      }

      xOpt = nn.weights();
    }

    return argument -> {
      Vec result = nn.apply(argument, xOpt);
      return ensemble.trans(result);
    };
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
