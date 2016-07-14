package com.spbsu.crawl.data.impl.system;

/**
 * Created by noxoomo on 13/07/16
 */
public class EmptyFieldsDefault {

  final public static int emptyInt() {
    return Integer.MIN_VALUE;
  }

  public static <T> T emptyValue() {
    return null;
  }

  public static boolean isEmpty(int value) {
    return value == emptyInt();
  }

  public static boolean isEmpty(long value) {
    return value == emptyLong();
  }

  public static <T> boolean isEmpty(T object) {
    return object == null;
  }

  public static <T> boolean notEmpty(T object) {
    return !isEmpty(object);
  }

  public static long emptyLong() {
    return Long.MIN_VALUE;
  }
}
