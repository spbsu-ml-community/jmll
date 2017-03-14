package com.spbsu.ml.cli.builders.methods;

import com.spbsu.commons.func.Factory;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.BFGrid;
import com.spbsu.ml.cli.builders.methods.impl.*;
import com.spbsu.ml.dynamicGrid.interfaces.DynamicGrid;
import com.spbsu.ml.methods.Optimization;
import com.spbsu.ml.methods.VecOptimization;

import java.lang.reflect.Method;
import java.util.StringTokenizer;

/**
 * User: qdeee
 * Date: 03.09.14
 */
public class MethodsBuilder {
  public void setGridBuilder(final Factory<BFGrid> gridFactory) {
    GreedyObliviousTreeBuilder.defaultGridBuilder = gridFactory;
    GreedyTDRegionBuilder.defaultGridBuilder = gridFactory;
    GreedyTDBumpyRegionBuilder.defaultGridBuilder = gridFactory;
    RegionForestBuilder.defaultGridBuilder = gridFactory;
    GreedyTDCherryRegionBuilder.defaultGridBuilder = gridFactory;
    RidgeRegressionLeavesObliviousTreeBuilder.defaultGridBuilder = gridFactory;
  }

  public void setDynamicGridBuilder(final Factory<DynamicGrid> dynamicGridFactory) {
    GreedyObliviousTreeDynamicBuilder.defaultDynamicGridBuilder = dynamicGridFactory;
    GreedyObliviousTreeDynamic2Builder.defaultDynamicGridBuilder = dynamicGridFactory;
  }

  public void setRandom(final FastRandom random) {
    BootstrapOptimizationBuilder.defaultRandom = random;
    RandomForestBuilder.defaultRandom = random;
    MultiClassSplitGradFacBuilder.defaultRandom = random;
    RidgeRegressionLeavesObliviousTreeBuilder.defaultRandom = random;
  }

  public VecOptimization create(final String scheme) {
    return chooseMethod(scheme);
  }

  private static VecOptimization chooseMethod(final String scheme) {
    final int parametersStart = scheme.indexOf('(') >= 0 ? scheme.indexOf('(') : scheme.length();
    final Factory<? extends VecOptimization> factory = methodBuilderByName(scheme.substring(0, parametersStart));
    final String parameters = parametersStart < scheme.length() ? scheme.substring(parametersStart + 1, scheme.lastIndexOf(')')) : "";
    final StringTokenizer paramsTok = new StringTokenizer(parameters, ",");
    final Method[] builderMethods = factory.getClass().getMethods();
    while (paramsTok.hasMoreTokens()) {
      final String param = paramsTok.nextToken();
      final int splitPos = param.indexOf('=');
      final String name = param.substring(0, splitPos).trim();

      final StringBuilder valueBuilder = new StringBuilder();
      {
        int open = 0;
        String token = param.substring(splitPos + 1, param.length()).trim();
        while (true) {
          valueBuilder.append(token);
          for (int i = 0; i < token.length(); ++i) {
            final char c = token.charAt(i);
            if (c == '(') {
              ++open;
            } else if (c == ')') {
              if (open <= 0) {
                throw new RuntimeException("Can not set up parameter \"" + name + "\" because of bad parsing stack");
              } else {
                --open;
              }
            }
          }
          if (open == 0) {
            break;
          } else {
            token = ',' + paramsTok.nextToken();
          }
        }
      }
      final String value = valueBuilder.toString();
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
        } else if (Boolean.class.equals(type) || boolean.class.equals(type)) {
          setter.invoke(factory, Boolean.parseBoolean(value));
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

  private static Factory<? extends VecOptimization> methodBuilderByName(final String name) {
    try {
      final Class<Factory<VecOptimization>> clazz = (Class<Factory<VecOptimization>>) Class.forName("com.spbsu.ml.cli.builders.methods.impl." + name + "Builder");
      return clazz.newInstance();
    } catch (ClassNotFoundException | InstantiationException | IllegalAccessException e) {
      throw new RuntimeException("Couldn't create weak model: " + name, e);
    }
  }
}
