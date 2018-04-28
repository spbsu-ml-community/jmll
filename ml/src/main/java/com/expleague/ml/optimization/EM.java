package com.expleague.ml.optimization;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.ml.func.LogLikelihood;
import com.expleague.ml.func.ReguralizerFunc;
import com.expleague.ml.models.mixture.Mixture;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.PrintStream;

public class EM implements Optimize<LogLikelihood> {
  private final int numIterations;
  private final Mx samples;
  private final int numSamples;
  private final Mixture mixture;
  private PrintStream debug = System.out;
  private Vec x;

  public EM(Mx samples, Mixture mixture, int numIterations, PrintStream debug) {
    this(samples, mixture, numIterations);
    this.debug = debug;
  }

  public EM(Mx samples, Mixture mixture, int numIterations) {
    this.numIterations = numIterations;
    this.samples = samples;
    this.numSamples = samples.rows();
    this.mixture = mixture;
    x = mixture.getParameters();
  }

  @Override
  public Vec optimize(LogLikelihood ll, ReguralizerFunc reg, Vec x0) {
    throw new NotImplementedException();
  }

  @Override
  public Vec optimize(LogLikelihood ll, Vec x0) {
    VecTools.assign(x, x0);
    return optimize(ll);
  }

  @Override
  public Vec optimize(LogLikelihood likelihood) {
    final Mx samplesByCompProb = new VecBasedMx(numSamples, mixture.numComponents());
    final Vec samplesProb = new ArrayVec(numSamples);
    double prevValue = Double.MAX_VALUE;

    for (int iter = 0; iter < numIterations; iter++) {
      mixture.probAll(samples, samplesProb);
      mixture.byComponentProb(samples, samplesByCompProb);

      final double value = likelihood.value(samplesProb) / numSamples;
      if (Math.abs(value - prevValue) < 1e-10) {
        break;
      }
      prevValue = value;

      debug.printf("iter [%d], ll(x|Θ) = %f\n", iter, -value);
      mixture.upgradeParameters(samplesByCompProb, samples);
    }

    return x;
  }
}
