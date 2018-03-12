package com.expleague.ml.models.nn;

import com.expleague.commons.math.MathTools;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class ConvNet extends NeuralSpider<Double, Vec> {
  private NodeCalcer[] nodeCalcers;
  private final int inputDim;
  private final int numParameters;
  private final int outputDim;
  private final double dropout;
  private final FastRandom rng = new FastRandom();

  public ConvNet(List<NodeCalcer> nodeCalcers, int inputDim, int numParameters,
                 int outputDim, double dropout) {
    this.nodeCalcers = nodeCalcers.toArray(new NodeCalcer[nodeCalcers.size()]);
    this.inputDim = inputDim;
    this.numParameters = numParameters;
    this.outputDim = outputDim;
    this.dropout = dropout;
  }

  @Override
  public int numParameters() {
    return numParameters;
  }

  @Override
  protected Topology topology(boolean dropout) {
    return new Topology.Stub() {
      @Override
      public int outputCount() {
        return outputDim;
      }

      @Override
      public boolean isDroppedOut(int nodeIndex) {
        //noinspection SimplifiableIfStatement
        if (!dropout || nodeIndex > nodeCalcers.length)
          return false;
        return ConvNet.this.dropout > MathTools.EPSILON && rng.nextDouble() < ConvNet.this.dropout;
      }

      @Override
      public int dim() {
        return inputDim;
      }

      @Override
      public Stream<NodeCalcer> stream() {
        return Arrays.stream(nodeCalcers);
      }

      @Override
      public NodeCalcer at(int i) {
        return nodeCalcers[i];
      }

      @Override
      public int length() {
        return nodeCalcers.length;
      }
    };
  }

  @Override
  public int xdim() {
    return inputDim;
  }
}
