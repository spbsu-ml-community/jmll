package com.spbsu.exp.dl.dnn.config;

import org.junit.Test;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.spbsu.exp.dl.dnn.rectifiers.RectifierType;

import org.junit.Assert;

/**
 * jmll
 *
 * @author ksenon
 */
public class LayerConfigTest extends Assert {

  private static final double DELTA = 1e-14;

  @Test
  public void testSerialize() throws Exception {
    final LayerConfig config = new LayerConfig();
    config.inputSize = 10;
    config.outputSize = 20;
    config.rectifierType = RectifierType.SIGMOID;
    config.bias = 0.1;
    config.bias_b = 0.;
    config.dropoutFraction = 0.;
    config.isTrain = true;

    final GsonBuilder builder = new GsonBuilder();
    builder.setPrettyPrinting();

    final Gson gson = builder.create();
    final String json = gson.toJson(config);

    final LayerConfig config_new = gson.fromJson(json, LayerConfig.class);

    assertEquals(config.inputSize, config_new.inputSize);
    assertEquals(config.outputSize, config_new.outputSize);
    assertEquals(config.rectifierType, config_new.rectifierType);
    assertEquals(config.bias, config_new.bias, DELTA);
    assertEquals(config.bias_b, config_new.bias_b, DELTA);
    assertEquals(config.dropoutFraction, config_new.dropoutFraction, DELTA);
    assertEquals(config.isTrain, config_new.isTrain);
  }

}
