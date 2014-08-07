package com.spbsu.exp.tools;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;

/**
 * User: qdeee
 * Date: 04.08.14
 */
public class Runner {
  public static void main(String[] args) {
    if (args.length == 0) {
      throw new IllegalArgumentException("Usage: java -jar exp.jar <local_package.runner_class> <args...>");
    }

    final String className = args[0];
    try {
      final Class<?> runClass = Class.forName("com.spbsu.exp." + className);
      final Method mainMethod = runClass.getDeclaredMethod("main", String[].class);
      final String[] methodArgs = Arrays.copyOfRange(args, 1, args.length);
      mainMethod.invoke(null, (Object)methodArgs);
    } catch (ClassNotFoundException | IllegalAccessException | NoSuchMethodException | InvocationTargetException e) {
      throw new RuntimeException("Something was wrong with calling 'main' method of " + className,
          e.fillInStackTrace());
    }
  }
}
