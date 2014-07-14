package com.spbsu.ml.data.tools;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;


import com.spbsu.ml.Func;
import com.spbsu.ml.data.set.DataSet;
import com.spbsu.ml.loss.L2;

/**
 * User: solar
 * Date: 14.07.14
 * Time: 20:32
 */
public class XXX<I> {
  public <T extends Func> T target(Class<T> targetClass) {
    try {
      return targetClass.newInstance();
    } catch ( InstantiationException | IllegalAccessException e) {
      throw new RuntimeException("Unable to create " + targetClass.getName() + " target");
    }
  }

  public static void main(String[] args) {
    L2 l = new XXX<>().target(L2.class);
  }
}
