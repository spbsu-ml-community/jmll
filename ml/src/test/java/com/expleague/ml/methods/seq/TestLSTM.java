package com.expleague.ml.methods.seq;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.ml.methods.seq.nn.LSTMLayer;
import com.expleague.ml.methods.seq.nn.LSTMNode;
import com.expleague.ml.methods.seq.nn.NetworkNode;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.ml.methods.seq.nn.NetworkLayer;
import org.junit.Assert;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class TestLSTM {
  private FastRandom random = new FastRandom(1);

  @Test
  public void testNodeGradient() {
    final int size = 10;
    final NetworkNode node = new LSTMNode(size, random);
    final Vec input = new ArrayVec(size + 2); // + 2 for previous cell state and output
    for (int i = 0; i < size + 2; i++) {
      input.set(i, random.nextDouble());
    }

    final Vec outputGrad = new ArrayVec(random.nextDouble(), random.nextDouble());
    final Vec output = node.value(input);
    final Vec paramsGrad = node.grad(input, outputGrad).gradByParams;
    final Vec params = node.params();
    final double eps = 1e-6;

    for (int i = 0; i < params.dim(); i++) {
      params.adjust(i, eps);
      final Vec newOutput = node.value(input);
      params.adjust(i, -eps);

      Assert.assertEquals((VecTools.multiply(outputGrad, newOutput) - VecTools.multiply(outputGrad, output)) / eps, paramsGrad.get(i), 1e-6);
    }

    final Vec inputGrad = node.grad(input, outputGrad).gradByInput;
    for (int i = 0; i < 2; i++) {
      input.adjust(i + size, eps);
      final Vec newOutput = node.value(input);
      input.adjust(i + size, -eps);
      //System.out.println((VecTools.multiply(outputGrad, newOutput) - VecTools.multiply(outputGrad, output)) / eps + " " + inputGrad.get(i));
      assertEquals((VecTools.multiply(outputGrad, newOutput) - VecTools.multiply(outputGrad, output)) / eps, inputGrad.get(i), 1e-6);
    }
  }

  @Test
  public void testLayerGradient() {
    final int nodeSize = 10;
    final int nodesCount = 25;
    final int inputSize = 15;
    final NetworkLayer layer = new LSTMLayer(nodesCount, nodeSize, random);
    Mx input = new VecBasedMx(inputSize, nodeSize);
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
