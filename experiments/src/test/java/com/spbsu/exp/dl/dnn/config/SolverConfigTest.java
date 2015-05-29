package com.spbsu.exp.dl.dnn.config;

import org.junit.Test;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.spbsu.exp.dl.GraphAdapterBuilder;
import com.spbsu.exp.dl.dnn.rectifiers.RectifierType;

import org.junit.Assert;

/**
 * jmll
 *
 * @author ksenon
 */
public class SolverConfigTest extends Assert {

  private static final double DELTA = 1e-14;

  @Test
  public void testSerialize() throws Exception {
    final NetConfig netConfig = new NetConfig();
    netConfig.input = 10;
    netConfig.output = 20;

    final SolverConfig solverConfig = new SolverConfig();
    solverConfig.batchSize = 1000;
    solverConfig.epochsNumber = 123;
    solverConfig.learningRate = 0.0001;
    solverConfig.net = netConfig;
    solverConfig.debug = true;

    final LayerConfig layerConfig_1 = new LayerConfig();
    layerConfig_1.inputSize = 10;
    layerConfig_1.outputSize = 40;
    layerConfig_1.rectifierType = RectifierType.SIGMOID;
    layerConfig_1.bias = 0.1;
    layerConfig_1.bias_b = 0.;
    layerConfig_1.dropoutFraction = 0.;
    layerConfig_1.isTrain = true;

    final LayerConfig layerConfig_2 = new LayerConfig();
    layerConfig_2.inputSize = 40;
    layerConfig_2.outputSize = 20;
    layerConfig_2.rectifierType = RectifierType.FLAT;
    layerConfig_2.bias = 0.4;
    layerConfig_2.bias_b = 1.;
    layerConfig_2.dropoutFraction = 5.;
    layerConfig_2.isTrain = true;

    netConfig.layers = new LayerConfig[]{layerConfig_1, layerConfig_2};

    final GsonBuilder builder = new GsonBuilder();
    builder.setPrettyPrinting();

    new GraphAdapterBuilder()
        .addType(NetConfig.class)
        .addType(LayerConfig.class)
        .registerOn(builder)
    ;

    final Gson gson = builder.create();
    final String json = gson.toJson(solverConfig);

    final SolverConfig solverConfig_new = gson.fromJson(json, SolverConfig.class);

    assertEquals(solverConfig.batchSize, solverConfig_new.batchSize);
    assertEquals(solverConfig.epochsNumber, solverConfig_new.epochsNumber);
    assertEquals(solverConfig.learningRate, solverConfig_new.learningRate, DELTA);
    assertEquals(solverConfig.debug, solverConfig_new.debug);

    final NetConfig netConfig_new = solverConfig_new.net;

    assertEquals(netConfig.input, netConfig_new.input);
    assertEquals(netConfig.output, netConfig_new.output);
    assertEquals(netConfig.layers.length, netConfig_new.layers.length);

    for (int i = 0; i < 2; i++) {
      final LayerConfig layerConfig_new = netConfig.layers[i];

      assertEquals(netConfig.layers[i].inputSize, layerConfig_new.inputSize);
      assertEquals(netConfig.layers[i].outputSize, layerConfig_new.outputSize);
      assertEquals(netConfig.layers[i].rectifierType, layerConfig_new.rectifierType);
      assertEquals(netConfig.layers[i].bias, layerConfig_new.bias, DELTA);
      assertEquals(netConfig.layers[i].bias_b, layerConfig_new.bias_b, DELTA);
      assertEquals(netConfig.layers[i].dropoutFraction, layerConfig_new.dropoutFraction, DELTA);
      assertEquals(netConfig.layers[i].isTrain, layerConfig_new.isTrain);
    }
  }

}
