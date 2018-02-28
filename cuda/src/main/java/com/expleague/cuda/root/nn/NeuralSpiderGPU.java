package com.expleague.cuda.root.nn;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.cuda.JCudaMemory;
import com.expleague.cuda.data.GPUVec;
import jcuda.driver.CUdeviceptr;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by hrundelb on 28.01.18.
 */
public class NeuralSpiderGPU {

  private GPUVec weights;

  private CUdeviceptr topology;

  private int topSize;

  private int argsSize;

  private GPUVec outStates;


  public NeuralSpiderGPU(Vec weights, int[] topology, int argsSize) {
    this.topology = JCudaMemory.allocInt(topology.length);
    JCudaMemory.copy(topology, this.topology);

    this.weights = new GPUVec(weights.toArray());
    this.topSize = topology.length / 3;
    this.argsSize = argsSize;
    this.outStates = new GPUVec(argsSize + topSize);
  }


  public GPUVec compute2(GPUVec args) {
    NetworkOperations.produceState2(args, argsSize, weights, topology, topSize, outStates);
    return outStates;
  }

  public GPUVec compute3(GPUVec args) {
    NetworkOperations.produceState3(args, argsSize, weights, topology, topSize, outStates);
    return outStates;
  }

  public static class TopologyBuilder {

    private List<Integer> topology = new ArrayList<>();

    private int weightsStart = 0;


    public TopologyBuilder addFullyConnectedLayer(int from, int to, int size) {
      int wCount = to - from;
      if (wCount < 1 || from < 0)
        throw new IllegalArgumentException();

      for (int i = 0; i < size; i++) {
        topology.add(from);
        topology.add(to);
        topology.add(weightsStart);
        weightsStart += wCount;
      }
      return this;
    }

    public int[] build() {
      return topology.stream().mapToInt(i -> i).toArray();
    }

    public int getWeightsCount() {
      return weightsStart;
    }
  }
}
