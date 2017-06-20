package com.spbsu.ml.methods.seq;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.methods.seq.nn.*;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestLogistic {
  private FastRandom random = new FastRandom(2);

  @Test
  public void testNodeGradient() {
    final int size = 100;
    final NetworkNode node = new LogisticNode(size, random);
    final Vec input = new ArrayVec(size);
    for (int i = 0; i < size; i++) {
      input.set(i, random.nextDouble());
    }

    final Vec outputGrad = new ArrayVec(random.nextDouble());
    final Vec output = node.value(input);
    final Vec paramsGrad = node.grad(input, outputGrad).gradByParams;
    final Vec params = node.params();
    final double eps = 1e-6;

    for (int i = 0; i < params.dim(); i++) {
      params.adjust(i, eps);
      final Vec newOutput = node.value(input);
      params.adjust(i, -eps);

      assertEquals((VecTools.multiply(outputGrad, newOutput) - VecTools.multiply(outputGrad, output)) / eps, paramsGrad.get(i), 1e-6);
    }

    final Vec inputGrad = node.grad(input, outputGrad).gradByInput;
    for (int i = 0; i < input.dim(); i++) {
      input.adjust(i, eps);
      final Vec newOutput = node.value(input);
      input.adjust(i, -eps);
      //System.out.println((VecTools.multiply(outputGrad, newOutput) - VecTools.multiply(outputGrad, output)) / eps + " " + inputGrad.get(i));
      assertEquals((VecTools.multiply(outputGrad, newOutput) - VecTools.multiply(outputGrad, output)) / eps, inputGrad.get(i), 1e-6);
    }
  }

  @Test
  public void testLayerGradient() {
    final int inputDim = 100;
    final int nodesCount = 25;
    final int inputSize = 150;
    final NetworkLayer layer = new LogisticLayer(nodesCount, inputDim, random);
    Mx input = new VecBasedMx(inputSize, inputDim);
    for (int i = 0; i < input.rows(); i++) {
      for (int j = 0; j < input.columns(); j++) {
        input.set(i, j, random.nextDouble());
      }
    }

    final Mx output = layer.value(input);
    final Mx outputGrad = new VecBasedMx(output.rows(), output.columns());
    for (int i = 0 ; i < outputGrad.rows(); i++) {
      for (int j = 0; j < outputGrad.columns(); j++) {
        outputGrad.set(i, j, random.nextDouble());
      }
    }

    final Vec grad = layer.gradient(input, outputGrad, true).gradByParams;
    final double eps = 1e-6;
    for (int i = 0; i < layer.paramCount(); i++) {
      final Vec dW = new ArrayVec(layer.paramCount());
      dW.set(i, eps);
      layer.adjustParams(dW);
      final Mx newOutput = layer.value(input);
      dW.set(i, -eps);
      layer.adjustParams(dW);

      //System.out.println((VecTools.multiply(newOutput, outputGrad) - VecTools.multiply(output, outputGrad)) / eps + " " + grad.get(i));
      assertEquals((VecTools.multiply(newOutput, outputGrad) - VecTools.multiply(output, outputGrad)) / eps, grad.get(i), 1e-5);
    }
  }
}
