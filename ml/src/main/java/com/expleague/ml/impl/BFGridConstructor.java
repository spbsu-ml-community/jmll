package com.expleague.ml.impl;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.BFGrid;
import com.expleague.ml.data.impl.BinarizedDataSet;
import org.jetbrains.annotations.Nullable;

import java.util.ArrayList;
import java.util.List;

public class BFGridConstructor implements BFGrid {
  private BFGridImpl delegate;

  public BFGrid.Row row(final int feature) {
    return build().row(feature);
  }

  public BFGrid.Feature bf(final int bfIndex) {
    return build().bf(bfIndex);
  }

  public void binarizeTo(final Vec x, final byte[] folds) {
    build().binarizeTo(x, folds);
  }

  public int size() {
    return build().size();
  }

  public int rows() {
    return build().rows();
  }

  @Nullable
  public BFGrid build() {
    if (delegate == null && features.size() > 0) {
      BFRowImpl[] rows = new BFRowImpl[features.size()];
      int bfindex = 0;
      for (int f = 0; f < rows.length; f++) {
        final int bfStart = bfindex;
        final List<ConstructorFeature> row = this.features.get(f);
        final double[] borders = row.stream().mapToDouble(Feature::condition).toArray();
        bfindex += row.size();
        rows[f] = new BFRowImpl(this, bfStart, f, borders);
        for (int b = 0; b < row.size(); b++) {
          row.get(b).binaryFeature = rows[f].bf(b);
        }
      }
      delegate = new BFGridImpl(rows);
    }
    return delegate;
  }

  List<List<ConstructorFeature>> features = new ArrayList<>();
  public Feature condition(int findex, double condition) {
    while (findex >= features.size()) {
      features.add(new ArrayList<>());
    }

    final List<ConstructorFeature> features = this.features.get(findex);
    int i = 0;
    for (; i < features.size(); i++) {
      final Feature next = features.get(i);
      if (next.condition() == condition) // equals here is not a mistake, deserialized conditions must be equal
        return next;

      if (next.condition() > condition)
        break;
    }
    final ConstructorFeature feature = new ConstructorFeature(findex, condition);
    features.add(i, feature);
    return feature;
  }

  static class ConstructorFeature implements Feature {
    private final int findex;
    private final double condition;


    private BinaryFeatureImpl binaryFeature;

    ConstructorFeature(int findex, double condition) {
      this.findex = findex;
      this.condition = condition;
    }

    public boolean value(final byte[] folds) {
      return binaryFeature.value(folds);
    }

    public boolean value(final byte fold) {
      return binaryFeature.value(fold);
    }

    public boolean value(final Vec vec) {
      return binaryFeature.value(vec);
    }

    public boolean value(int index, BinarizedDataSet bds) {
      return binaryFeature.value(index, bds);
    }

    public int findex() {
      return findex;
    }

    public int bin() {
      return binaryFeature.bin();
    }

    public int index() {
      return binaryFeature.index();
    }

    public double condition() {
      return condition;
    }

    public double power() {
      throw new UnsupportedOperationException();
    }

    public BFRowImpl row() {
      return binaryFeature.row();
    }
  }
}
