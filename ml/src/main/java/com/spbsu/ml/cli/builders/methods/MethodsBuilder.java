package com.spbsu.ml.cli.builders.methods;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.cli.builders.methods.impl.GreedyObliviousTreeBuilder;
import com.spbsu.ml.cli.builders.methods.impl.GreedyObliviousTreeDynamic2Builder;
import com.spbsu.ml.cli.builders.methods.impl.GreedyObliviousTreeDynamicBuilder;
import com.spbsu.ml.cli.builders.methods.impl.GreedyTDRegionBuilder;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.methods.Optimization;
import com.spbsu.ml.methods.VecOptimization;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Random;
import java.util.StringTokenizer;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class MethodsBuilder {
  private final Random random;

  public MethodsBuilder(final Random random) {
    this.random = random;
  }

  public void setGridBuilder(final Factory<BFGrid> gridFactory) {
    GreedyObliviousTreeBuilder.defaultGridBuilder = gridFactory;
    GreedyTDRegionBuilder.defaultGridBuilder = gridFactory;
  }

  public void setDynamicGridBuilder(final Factory<DynamicGrid> dynamicGridFactory) {
    GreedyObliviousTreeDynamicBuilder.defaultDynamicGridBuilder = dynamicGridFactory;
    GreedyObliviousTreeDynamic2Builder.defaultDynamicGridBuilder = dynamicGridFactory;
  }

  public VecOptimization create(final String scheme) {
    return chooseMethod(scheme);
  }

  private VecOptimization chooseMethod(String scheme) {
    final int parametersStart = scheme.indexOf('(') >= 0 ? scheme.indexOf('(') : scheme.length();
    final Factory<? extends VecOptimization> factory = methodBuilderByName(scheme.substring(0, parametersStart));
    final String parameters = parametersStart < scheme.length() ? scheme.substring(parametersStart + 1, scheme.lastIndexOf(')')) : "";
    final StringTokenizer paramsTok = new StringTokenizer(parameters, ",");
    final Method[] builderMethods = factory.getClass().getMethods();
    while (paramsTok.hasMoreTokens()) {
      final String param = paramsTok.nextToken();
      final int splitPos = param.indexOf('=');
      final String name = param.substring(0, splitPos).trim();
      final String value = param.substring(splitPos + 1, param.length()).trim();
      Method setter = null;
      for (int m = 0; m < builderMethods.length; m++) {
        if (builderMethods[m].getName().equalsIgnoreCase("set" + name)) {
          setter = builderMethods[m];
          break;
        }
      }
      if (setter == null || setter.getParameterTypes().length > 1 || setter.getParameterTypes().length < 1) {
        System.err.println("Can not set up parameter: " + name + " to value: " + value + ". No setter in builder.");
        continue;
      }
      final Class type = setter.getParameterTypes()[0];
      try {
        if (Integer.class.equals(type) || int.class.equals(type)) {
          setter.invoke(factory, Integer.parseInt(value));
        } else if (Double.class.equals(type) || double.class.equals(type)) {
          setter.invoke(factory, Double.parseDouble(value));
        } else if (String.class.equals(type)) {
          setter.invoke(factory, value);
        } else if (Optimization.class.isAssignableFrom(type)) {
          setter.invoke(factory, chooseMethod(value));
        } else {
          System.err.println("Can not set up parameter: " + name + " to value: " + value + ". Unknown parameter type: " + type + "");
        }
      } catch (Exception e) {
        throw new RuntimeException("Can not set up parameter: " + name + " to value: " + value + "", e);
      }
    }
    return factory.create();
  }

  private Factory<? extends VecOptimization> methodBuilderByName(final String name) {
    try {
      final Class<Factory<VecOptimization>> clazz = (Class<Factory<VecOptimization>>) Class.forName("com.spbsu.ml.cli.builders.methods.impl." + name + "Builder");
      final Factory<VecOptimization> builder = clazz.newInstance();
      final Method setRandom;
      try {
        setRandom = builder.getClass().getDeclaredMethod("setRandom", FastRandom.class);
        setRandom.invoke(builder, random);
        System.out.println("hop");
      } catch (NoSuchMethodException | InvocationTargetException ignored) {}
      return builder;
    } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
      throw new RuntimeException("Couldn't create weak model: " + name, e);
    }
  }
}
