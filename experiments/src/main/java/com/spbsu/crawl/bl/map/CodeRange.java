package com.spbsu.crawl.bl.map;

import java.util.Arrays;

/**
 * Created by noxoomo on 02/05/16.
 */
public class CodeRange {
  int[] codes;

  public CodeRange(int... codes) {
    this.codes = codes;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    CodeRange codeRange = (CodeRange) o;

    return Arrays.equals(codes, codeRange.codes);

  }

  boolean contains(int i) {
    for (int idx : codes) {
      if (idx == i) {
        return true;
      }
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(codes);
  }
}
