package com.expleague.ml;

import com.expleague.commons.func.Converter;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.ml.data.impl.BinarizedDataSet;
import com.expleague.ml.impl.BFRowImpl;
import com.expleague.ml.io.BFGridStringConverter;
import com.expleague.ml.meta.FeatureMeta;

public interface BFGrid {
  Converter<BFGrid, CharSequence> CONVERTER = new BFGridStringConverter();

  Row row(int feature);
  Feature bf(int bfIndex);
  void binarizeTo(Vec x, byte[] folds);

  int rows();
  int size();

  interface Row {
    int bin(double val);
    Feature bf(int index);
    double condition(int border);

    int findex();
    boolean ordered();

    int start();
    int end();
    int size();
    boolean empty();

    BFGrid grid();

    FeatureMeta fmeta();
  }

  interface Feature {
    boolean value(byte[] folds);
    boolean value(byte fold);
    boolean value(Vec vec);
    boolean value(int index, BinarizedDataSet bds);

    int findex();
    int bin();
    int index();

    double condition();
    double power();

    Row row();
  }
}
