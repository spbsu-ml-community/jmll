package com.spbsu.exp.dl.dnn;

import com.spbsu.exp.dl.GraphAdapterBuilder;
import org.jetbrains.annotations.NotNull;

import com.spbsu.exp.dl.dnn.config.NetConfig;
import com.spbsu.exp.dl.dnn.config.SolverConfig;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.exp.dl.dnn.config.LayerConfig;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * jmll
 *
 * @author ksenon
 */
public class NetBuilder {

  private final Gson gson;

  public NetBuilder() {
    final GsonBuilder builder = new GsonBuilder();

    new GraphAdapterBuilder()
        .addType(NetConfig.class)
        .addType(LayerConfig.class)
        .registerOn(builder)
    ;
    gson = builder.create();
  }

  public Solver buildSolver(final @NotNull File configuration) throws IOException {
    final SolverConfig solverConfig = gson.fromJson(
        new BufferedReader(new FileReader(configuration)), SolverConfig.class
    );

    final Solver solver = new Solver();
    solver.batchSize    = solverConfig.batchSize;
    solver.epochsNumber = solverConfig.epochsNumber;
    solver.learningRate = solverConfig.learningRate;
    solver.initMethod   = solverConfig.initMethod;
    solver.debug        = solverConfig.debug;
    solver.net          = buildNet(solverConfig.net, solver);

    return solver;
  }

  private FullyConnectedNet buildNet(final NetConfig netConfig, final Solver solver) {
    final FullyConnectedNet net = new FullyConnectedNet();
    net.input  = new VecBasedMx(solver.batchSize, netConfig.input);
    net.output = new VecBasedMx(solver.batchSize, netConfig.output);
    net.debug  = solver.debug;
    net.layers = buildLayers(netConfig.layers, solver);

    return net;
  }

  private Layer[] buildLayers(final LayerConfig[] layersConfigs, final Solver solver) {
    final Layer[] layers = new Layer[layersConfigs.length];

    for (int i = 0; i < layersConfigs.length; i++) {
      final LayerConfig layerConfig = layersConfigs[i];

      final Layer layer = new Layer();
      layer.input =           new VecBasedMx(solver.batchSize, layerConfig.inputSize);
      layer.output =          new VecBasedMx(solver.batchSize, layerConfig.outputSize);
      layer.activations =     new VecBasedMx(solver.batchSize, layerConfig.outputSize);
      layer.dropoutMask =     new VecBasedMx(solver.batchSize, layerConfig.outputSize);
      layer.weights =         new VecBasedMx(layerConfig.outputSize, layerConfig.inputSize);
      layer.difference =      new VecBasedMx(layerConfig.outputSize, layerConfig.inputSize);
      layer.rectifier =       layerConfig.rectifierType.getInstance();
      layer.bias =            layerConfig.bias;
      layer.bias_b =          layerConfig.bias_b;
      layer.dropoutFraction = layerConfig.dropoutFraction;
      layer.isTrain =         layerConfig.isTrain;
      layer.debug =           solver.debug;

      layers[i] = layer;
    }

    return layers;
  }

}
