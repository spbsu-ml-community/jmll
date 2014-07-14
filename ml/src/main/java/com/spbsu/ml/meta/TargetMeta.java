package com.spbsu.ml.meta;

/**
 * User: solar
 * Date: 07.07.14
 * Time: 15:53
 */
public interface TargetMeta {
  TargetMeta FAKE = new TargetMeta() {
    @Override
    public ValueType type() {
      return ValueType.REAL;
    }
  };

  ValueType type();
  enum ValueType {
    REAL,
  }
}
