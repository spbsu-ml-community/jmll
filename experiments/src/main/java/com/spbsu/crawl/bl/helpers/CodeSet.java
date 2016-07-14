package com.spbsu.crawl.bl.helpers;

import java.util.Arrays;

/**
 * Created by noxoomo on 02/05/16.
 */
public class CodeSet {
  int[] codes;

  public CodeSet(int... codes) {
    this.codes = codes;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;

    CodeSet codeSet = (CodeSet) o;

    return Arrays.equals(codes, codeSet.codes);

  }

  public boolean contains(int i) {
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
