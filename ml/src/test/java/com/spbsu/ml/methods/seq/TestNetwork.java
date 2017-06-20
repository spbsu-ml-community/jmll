package com.spbsu.ml.methods.seq;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.methods.seq.nn.LSTMLayer;
import com.spbsu.ml.methods.seq.nn.LogisticLayer;
import com.spbsu.ml.methods.seq.nn.MeanPoolLayer;
import com.spbsu.ml.methods.seq.nn.NeuralNetwork;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestNetwork {
  private final FastRandom random = new FastRandom(1);

  @Test
  public void testNetwork() {
    final int signalDim = 15;
    final int inputDim = 25;
    final int lstmNodeCount = 10;
    final int logisticNodeCount = 20;

    NeuralNetwork network = new NeuralNetwork(
            new LSTMLayer(lstmNodeCount, signalDim, random),
            new MeanPoolLayer(),
            new LogisticLayer(logisticNodeCount, lstmNodeCount, random)
    );

    final Mx input = new VecBasedMx(inputDim, signalDim);
    for (int i = 0; i < inputDim; i++) {
      for (int j= 0; j < signalDim; j++) {
        input.set(i, j, random.nextDouble());
      }
    }
    final Vec output = network.value(input);
    final Mx outputGrad = new VecBasedMx(1, logisticNodeCount);
    VecTools.fill(outputGrad, 1);

    final Vec grad = network.gradByParams(input, outputGrad, false);
    final double eps = 1e-6;

    for (int i = 0; i < network.paramCount(); i++) {
      final Vec dW = new ArrayVec(grad.dim());
      dW.set(i, eps);
      network.adjustParams(dW);
      final Vec newOutput = network.value(input);
      dW.set(i, -eps);
      network.adjustParams(dW);
      //System.out.println((VecTools.multiply(newOutput, outputGrad) - VecTools.multiply(output, outputGrad)) / eps + " " + grad.get(i));
      assertEquals((VecTools.multiply(newOutput, outputGrad) - VecTools.multiply(output, outputGrad)) / eps, grad.get(i), 1e-5);
    }
  }
}
