package com.spbsu.ml.models.gpf;

import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;

/**
 * User: irlab
 * Date: 16.05.14
 */
public class Tensor4 {
  public final ArrayVec vec;
  public final int dim1;
  public final int dim2;
  public final int dim3;
  public final int dim4;

  public Tensor4(int dim1, int dim2, int dim3, int dim4) {
    this.dim1 = dim1;
    this.dim2 = dim2;
    this.dim3 = dim3;
    this.dim4 = dim4;
    vec = new ArrayVec(dim1 * dim2 * dim3 * dim4);
  }

  private int index(int i, int j, int k, int l) {
    assert(0 <= i && i < dim1);
    assert(0 <= j && j < dim2);
    assert(0 <= k && k < dim3);
    assert(0 <= l && l < dim4);
    return dim4 * (dim3 * (dim2 * i + j) + k) + l;
  }

  public double get(int i, int j, int k, int l) {
    return vec.get(index(i, j, k, l));
  }

  public Tensor4 set(int i, int j, int k, int l, double val) {
    vec.set(index(i, j, k, l), val);
    return this;
  }

  public ArrayVec getRow(int i, int j, int k) {
    return vec.sub(index(i, j, k, 0), dim4);
  }

  public Tensor4 setRow(int i, int j, int k, Vec val) {
    if (val.dim() != dim4)
      throw new IllegalArgumentException("val.dim() != dim4, val.dim() = " + val.dim() + ", dim4 = " + dim4);
    int index = index(i, j, k, 0);
    for (int l = 0; l < dim4; l++)
      vec.set(index + l, val.get(l));
    return this;
  }

  public Tensor4 adjust(int i, int j, int k, int l, double increment) {
    vec.adjust(index(i, j, k, l), increment);
    return this;
  }

  public String toString() {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          for (int l = 0; l < dim4; l++) {
            builder.append(l > 0 ? "\t" : "");
            builder.append(get(i, j, k, l));
          }
          builder.append('\n');
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
    return (vec.hashCode() << 1) + dim2 * dim3 * dim4;
  }
}
