package com.expleague.cuda;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;
import com.expleague.commons.random.FastRandom;
import com.expleague.cuda.data.GPUVec;
import com.expleague.cuda.root.nn.NeuralSpiderGPU;
import com.expleague.ml.models.nn.LayeredNetwork;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Created by hrundelb on 28.01.18.
 */
public class LayeredNetworkTests {

  public static final double DELTA = 1e-5;

  @Test
  public void testValue() {
    Random random = new FastRandom();

    NeuralSpiderGPU.TopologyBuilder builder = new NeuralSpiderGPU.TopologyBuilder();
    int[] topology = builder.addFullyConnectedLayer(0, 4, 4)
        .addFullyConnectedLayer(4, 8, 3).addFullyConnectedLayer(8, 11, 1).build();
    System.out.println("Topology: " + Arrays.toString(topology));

    int weightsCount = builder.getWeightsCount();
    Vec weights2 = new ArrayVec(IntStream.range(-4, weightsCount).mapToDouble(i -> i < 0 ? 0 : random.nextDouble()).toArray());
    Vec weights = weights2.sub(4, weights2.dim() - 4);
    System.out.println("Weights: " + weights);
    System.out.println("Weights2: " + weights2);

    Vec args = new ArrayVec(2,3,1,4);
    NeuralSpiderGPU neuralSpiderGPU = new NeuralSpiderGPU(weights, topology, args.length());

    GPUVec vec = neuralSpiderGPU.compute2(new GPUVec(args.toArray()));
    System.out.println(Arrays.toString(vec.toArray()));
    System.out.println("======");

    GPUVec vec2 = neuralSpiderGPU.compute3(new GPUVec(args.toArray()));
    System.out.println(Arrays.toString(vec2.toArray()));
    System.out.println("======");

    LayeredNetwork nn = new LayeredNetwork(random, 0, 4, 4, 3, 1);
    final Vec vec3 = nn.compute(args, weights2);
    System.out.println(vec3);

    double result1 = DoubleStream.of(vec.toArray()).reduce((a, b) -> b).getAsDouble();
    double result2 = DoubleStream.of(vec2.toArray()).reduce((a, b) -> b).getAsDouble();
    double result3 = DoubleStream.of(vec3.toArray()).reduce((a, b) -> b).getAsDouble();
    Assert.assertEquals(result1, result3, DELTA);
    Assert.assertEquals(result2, result3, DELTA);
  }


  @Test
  public void testValue2Speed() {
    Random random = new FastRandom();
    int size = 250;
    int layers = 30;
    int iterations = 10;

    NeuralSpiderGPU.TopologyBuilder builder = new NeuralSpiderGPU.TopologyBuilder();
    int[] config = new int[layers + 1];
    for (int i = 0; i < layers; i++) {
      builder.addFullyConnectedLayer(i * size, (i + 1) * size, i == layers - 1 ? 1 : size);
      config[i] = size;
    }
    int[] topology = builder.build();
    config[config.length - 1] = 1;

    int weightsCount = builder.getWeightsCount();
    Vec weights2 = new ArrayVec(IntStream.range(-size, weightsCount).mapToDouble(i -> i < 0 ? 0 : random.nextGaussian()).toArray());
    Vec weights = weights2.sub(size, weights2.dim() - size);

    NeuralSpiderGPU neuralSpiderGPU = new NeuralSpiderGPU(weights, topology, size);
    LayeredNetwork layeredNetwork = new LayeredNetwork(random, 0, config);


    Vec[] args = new Vec[iterations];
    GPUVec[] gpuArgs = new GPUVec[iterations];
    for (int i = 0; i < iterations; i++) {
      double[] values = IntStream.range(0, size).mapToDouble(k -> random.nextGaussian()).toArray();
      args[i] = new ArrayVec(values);
      gpuArgs[i] = new GPUVec(values);
    }

    long startCPU = System.currentTimeMillis();
    Vec compute = null;
    for (int i = 0; i < iterations; i++) {
       compute = layeredNetwork.compute(args[i], weights2);
    }
    long timeCPU = System.currentTimeMillis() - startCPU;
    System.out.println("TimeCPU: " + timeCPU + " ms");


    long startGPU = System.currentTimeMillis();
    GPUVec gpuCompute = null;
    for (int i = 0; i < iterations; i++) {
      gpuCompute = neuralSpiderGPU.compute2(gpuArgs[i]);
    }
    long timeGPU = System.currentTimeMillis() - startGPU;
    System.out.println("TimeGPU: " + timeGPU + " ms");

    double result = compute.get(0);
    double gpuResult = DoubleStream.of(gpuCompute.toArray()).reduce((a, b) -> b).getAsDouble();
    System.out.println("Iterations: " + iterations + ", size: " + size + ", layers: " + layers);
    System.out.println("Result: " + result);
    System.out.println("GPUResult: " + gpuResult);
    Assert.assertEquals(result, gpuResult, DELTA);
  }
}
