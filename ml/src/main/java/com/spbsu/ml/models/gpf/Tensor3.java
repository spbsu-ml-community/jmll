package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.util.ArrayTools;

/**
 * Created with IntelliJ IDEA.
 * User: irlab
 * Date: 16.05.14
 * Time: 17:27
 * To change this template use File | Settings | File Templates.
 */
public class Tensor3 {
  public final ArrayVec vec;
  public final int dim1;
  public final int dim2;
  public final int dim3;

  public Tensor3(int dim1, int dim2, int dim3) {
    this.dim1 = dim1;
    this.dim2 = dim2;
    this.dim3 = dim3;
    vec = new ArrayVec(dim1 * dim2 * dim3);
  }

  private int index(int i, int j, int k) {
    assert(0 <= i && i < dim1);
    assert(0 <= j && j < dim2);
    assert(0 <= k && k < dim3);
    return dim3 * (dim2 * i + j) + k;
  }

  public double get(int i, int j, int k) {
    return vec.get(index(i, j, k));
  }

  public Tensor3 set(int i, int j, int k, double val) {
    vec.set(index(i, j, k), val);
    return this;
  }

  public ArrayVec getRow(int i, int j) {
    return vec.sub(index(i, j, 0), dim3);
  }

  public Tensor3 setRow(int i, int j, Vec val) {
    if (val.dim() != dim3)
      throw new IllegalArgumentException("val.dim() != dim3, val.dim() = " + val.dim() + ", dim3 = " + dim3);
    int index = index(i, j, 0);
    for (int l = 0; l < dim3; l++)
      vec.set(index + l, val.get(l));
    return this;
  }

  public Tensor3 adjust(int i, int j, int k, double increment) {
    vec.adjust(index(i, j, k), increment);
    return this;
  }

  public String toString() {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          builder.append(k > 0 ? "\t" : "");
          builder.append(get(i, j, k));
        }
        builder.append('\n');
      }
      builder.append('\n');
    }
    return builder.toString();
  }

  public double[] toArray() {
    return vec.toArray();
  }

  public boolean equals(Object o) {
    return o instanceof Tensor3 && (((Tensor3)o).dim1 == dim1) && (((Tensor3)o).dim2 == dim2) && ((Tensor3)o).vec.equals(vec);
  }

  public int hashCode() {
    return (vec.hashCode() << 1) + dim2 * dim3;
  }
}
