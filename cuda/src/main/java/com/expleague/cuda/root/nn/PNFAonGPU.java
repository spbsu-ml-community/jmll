package com.expleague.cuda.root.nn;


import com.expleague.commons.func.Computable;
import com.expleague.commons.math.FuncC1;
import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.SingleValueVec;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.seq.IntSeq;
import com.expleague.commons.seq.Seq;
import com.expleague.cuda.DeviceOperations;
import com.expleague.cuda.KernelOperations;
import com.expleague.cuda.data.GPUMx;
import com.expleague.cuda.data.GPUVec;
import com.expleague.cuda.root.array.VectorScale;
import com.expleague.ml.data.set.DataSet;
import com.expleague.ml.func.FuncEnsemble;
import com.expleague.ml.loss.WeightedL2;
import com.expleague.ml.methods.SeqOptimization;
import com.expleague.ml.methods.seq.PNFA;
import com.expleague.ml.optimization.Optimize;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by hrundelb on 05.09.17.
 */
public class PNFAonGPU<Loss extends WeightedL2> implements SeqOptimization<Integer, Loss> {
  private final int stateCount;
  private final int alphabetSize;
  private final Random random;
  private final Optimize<FuncEnsemble<? extends FuncC1>> optimize;
  private static final double lambda = -0.001;

  //value
  final Vec distribution;
  //final Vec distributionRes;
  final GPUMx w;

  //private final PNFA<Loss> pnfa;


  public PNFAonGPU(final int stateCount, final int alphabetSize, final Random random, final Optimize<FuncEnsemble<? extends FuncC1>> optimize) {
    this.stateCount = stateCount;
    this.alphabetSize = alphabetSize;
    this.random = random;
    this.optimize = optimize;

    //vale
    this.distribution = new GPUVec(stateCount);
    //this.distributionRes = new gpuVec(stateCount);
    this.w = new GPUMx(stateCount, stateCount);

    //this.pnfa = new PNFA<>(stateCount, alphabetSize, random, optimize);
  }

  @Override
  public Computable<Seq<Integer>, Vec> fit(final DataSet<Seq<Integer>> learn, final Loss loss) {
    final Vec params = init(loss.target);
    FuncC1[] funcs = new FuncC1[learn.length()];
    for (int i = 0; i < learn.length(); i++) {
      final IntSeq seq = (IntSeq) learn.at(i);
      double y = loss.target.get(i);
      double weight = loss.getWeights().get(i);
      //FuncC1 gpuFunc = new PNFAonGPU
      //    .PNFAPointLossFunc(seq, y, weight);
      //FuncC1 cpuFunc = pnfa.new PNFAPointLossFunc(seq, y, weight);
      funcs[i] = new PNFAPointLossFunc(seq, y, weight);
    }

    final Vec optParams = optimize.optimize(new FuncEnsemble<>(Arrays.asList(funcs), 1), params);
    return (seq) -> new SingleValueVec(getSeqValue(optParams, (IntSeq) seq));
  }

  private Vec init(Vec target) {
    final Vec params = new ArrayVec(stateCount * (stateCount - 1) * alphabetSize + stateCount);
    for (int c = 0; c < alphabetSize; c++) {

      final int mxSize = stateCount * (stateCount - 1);
      final Mx beta = new VecBasedMx(stateCount - 1, params.sub(c * mxSize, mxSize));
      for (int i = 0; i < stateCount; i++) {
        for (int j = 0; j < stateCount - 1; j++) {
          beta.set(i, j, random.nextGaussian());
        }
        if (i < stateCount - 1) {
          beta.adjust(i, i, stateCount / 2.0 + 3); // TODO change it
        }
      }
      for (int i = 0; i < stateCount - 1; i++) {
        beta.adjust(stateCount - 1, i, -stateCount / 2.0 - 3); // TODO change it
      }
    }

    final Vec values = getValues(params);
    final double[] targetValues = target.toArray();
    Arrays.sort(targetValues);
    for (int i = 0; i < stateCount; i++) {
      values.set(i, targetValues[(int) ((i + 0.5) * target.dim() / stateCount)]);
    }

    return params;
  }

  private GPUMx getMx(final Vec params, final int c) {
    final int mxSize = stateCount * (stateCount - 1);
    return new GPUMx(stateCount, params.sub(c * mxSize, mxSize));
  }

  private void getWeightMx(final Vec params, final int c, final GPUMx result) {
    final GPUMx beta = getMx(params, c);
    KernelOperations.fMatrixExp(beta, result);
  }

  private Vec getValues(final Vec params) {
    return params.sub(params.dim() - stateCount, stateCount);
  }

  private double getSeqValue(final Vec params, final IntSeq seq) {
    VecTools.fill(distribution, 1.0 / stateCount);
    for (int i = 0; i < seq.length(); i++) {
      getWeightMx(params, seq.intAt(i), w);
      multiplyLeft(distribution, w, distribution);
    }
    double multiply = VecTools.multiply(distribution, getValues(params));
    return multiply;
  }

  public class PNFAPointLossFunc extends FuncC1.Stub {

    private final IntSeq seq;
    private final double y;
    private final double weight;
    private final int[] seqAlphabet;
    private final TIntIntMap alphabetToOrderMap = new TIntIntHashMap();

    private final GPUVec gradientResult;

    //gradient
    final GPUMx[] betaGrad;
    final GPUMx[] ws;
    final Vec[] distributions;
    final Vec expectedTmp;

    public PNFAPointLossFunc(final IntSeq seq, final double y, final double weight) {
      this.seq = seq;
      this.y = y;
      this.weight = weight;

      this.seqAlphabet = seq.stream().sorted().distinct().toArray();
      for (int i = 0; i < seqAlphabet.length; i++) {
        alphabetToOrderMap.put(seqAlphabet[i], i);
      }
      this.gradientResult = new GPUVec(dim());

      //gradient
      this.betaGrad = new GPUMx[seqAlphabet.length];
      this.ws = new GPUMx[seqAlphabet.length];
      int n = stateCount * (stateCount - 1);
      for (int i = 0; i < seqAlphabet.length; i++) {
        betaGrad[i] = new GPUMx(stateCount, gradientResult.sub(seqAlphabet[i] * n, n));
        ws[i] = new GPUMx(stateCount, stateCount);
      }
      this.distributions = new Vec[seq.length() + 1];
      for (int i = 0; i < distributions.length; i++) {
        distributions[i] = new GPUVec(stateCount);
      }
      this.expectedTmp = new GPUVec(stateCount);
    }

    @Override
    public int dim() {
      return stateCount * (stateCount - 1) * alphabetSize + stateCount;
    }

    @Override
    public double value(Vec x) {
      return weight * MathTools.sqr(getSeqValue(x, seq) - y);
    }

    @Override
    public Vec gradient(Vec x) {
      VecTools.fill(gradientResult, 0.);
      for (int i = 0; i < seqAlphabet.length; i++) {
        getWeightMx(x, seqAlphabet[i], ws[i]);
      }

      VecTools.fill(distributions[0], 1.0 / stateCount);

      for (int i = 0; i < seq.length(); i++) {
        multiplyLeft(distributions[i], ws[alphabetToOrderMap.get(seq.intAt(i))],
            distributions[i + 1]);
      }

      Vec expectedValue = getValues(x);

      final double diff = VecTools.multiply(distributions[seq.length()], expectedValue) - y;

      for (int i = seq.length() - 1; i >= 0; i--) {
        final int a = alphabetToOrderMap.get(seq.intAt(i));

        for (int to = 0; to < stateCount; to++) {

          KernelOperations.fMatrixKernel1((float)weight, (float)diff, (GPUVec) distributions[i],
              (GPUVec) expectedValue, betaGrad[a], to, ws[a]);
        }
        multiply(ws[a], expectedValue, expectedTmp);
        expectedValue = expectedTmp;
      }

      for (int i = 0; i < seqAlphabet.length; i++) {
        for (int to = 0; to < stateCount; to++) {

          KernelOperations.fMatrixKernel2((float)lambda, betaGrad[i], to, ws[i]);
        }
      }

      return gradientResult;
    }
  }

  public static void multiplyLeft(Vec vec, GPUMx mx, Vec result) {
    final GPUMx resultMx = new GPUMx(1, result);
    DeviceOperations.multiply(new GPUMx(1, vec), mx, resultMx);
  }

  public static void multiply(Mx mx, Vec vec, Vec result) {
    GPUMx mxResult = new GPUMx(mx.rows(), result);
    DeviceOperations.multiply((GPUMx) mx, new GPUMx(mx.columns(), vec), mxResult);
  }

  public class LossFuncCPUandGPU extends FuncC1.Stub {

    private FuncC1 cpuFunc;
    private FuncC1 gpuFunc;
    private double delta = 0.00001;

    public LossFuncCPUandGPU(FuncC1 cpuFunc, FuncC1 gpuFunc) {
      this.cpuFunc = cpuFunc;
      this.gpuFunc = gpuFunc;
    }

    @Override
    public double value(Vec x) {
      double cpuValue = cpuFunc.value(new ArrayVec(x.toArray()));
      double gpuValue = gpuFunc.value(x);
      //System.out.println(String.format("Value call - cpu: %s - gpu: %s", cpuValue, gpuValue));
      checkEqual(cpuValue, gpuValue);
      return gpuValue;
    }


    @Override
    public Vec gradient(Vec x) {
      Vec cpuGradient = cpuFunc.gradient(new ArrayVec(x.toArray()));
      Vec gpuGradient = gpuFunc.gradient(x);
      double[] cpu = cpuGradient.toArray();
      double[] gpu = gpuGradient.toArray();
      //System.out.println(String.format("Gradient call\n- cpu: %s\n- gpu: %s",
      //    Arrays.toString(cpu), Arrays.toString(gpu)));
      checkEqual(cpu, gpu);
      return gpuGradient;
    }

    @Override
    public int dim() {
      int cpuDim = cpuFunc.dim();
      int gpuDim = gpuFunc.dim();
      checkEqual(cpuDim, gpuDim);
      return gpuDim;
    }

    private void checkEqual(double[] cpu, double[] gpu) {
      for (int i = 0; i < cpu.length; i++) {
        checkEqual(cpu[i], gpu[i]);
      }
    }


    private void checkEqual(double cpuValue, double gpuValue) {
      if (Math.abs(cpuValue - gpuValue) > delta) {
        throw new RuntimeException(String.format("%s not equal to %s with delta %s!", cpuValue,
            gpuValue, delta));
      }
    }
  }
}