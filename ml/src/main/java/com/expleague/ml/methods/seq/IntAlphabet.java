package com.expleague.ml.methods.seq;

import com.expleague.commons.seq.Seq;
import com.expleague.commons.seq.regexp.Alphabet;
import com.expleague.commons.seq.regexp.Matcher;

public class IntAlphabet implements Alphabet<Integer> {
  private final int size;

  public IntAlphabet(int size) {
    this.size = size;
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int indexCondition(Matcher.Condition<Integer> c) {
    if (!(c instanceof IntCondition)) {
      throw new IllegalArgumentException("Not a int condition");
    }
    return ((IntCondition) c).my;
  }

  @Override
  public Matcher.Condition<Integer> condition(int i) {
    return new IntCondition(i);
  }

  @Override
  public Matcher.Condition<Integer> conditionByT(Integer i) {
    return new IntCondition(i);
  }

  @Override
  public Integer getT(Matcher.Condition<Integer> condition) {
    if (condition instanceof IntCondition) {
      return ((IntCondition) condition).my;
    }
    return null;
  }

  @Override
  public int index(Integer integer) {
    return integer;
  }

  @Override
  public int index(Seq<Integer> seq, int index) {
    return seq.at(index);
  }

  private class IntCondition implements Matcher.Condition<Integer> {
    private final int my;

    IntCondition(int my) {
      this.my = my;
    }

    @Override
    public boolean is(Integer frag) {
      return frag == my;
    }

    @Override
    public String toString() {
      return Integer.toString(my);
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (o == null || !(o instanceof IntCondition)) {
        return false;
      }

      final IntCondition c = (IntCondition) o;
      return c.my == my;
    }

    @Override
    public int hashCode() {
      return Integer.hashCode(my);
    }
  }
}
