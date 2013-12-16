package com.spbsu.ml;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.ArrayVec;
import com.spbsu.commons.math.vectors.impl.VecBasedMx;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public interface Trans {
  int xdim();
  int ydim();
  @Nullable
  Trans gradient();
  Vec trans(Vec x);
  Mx transAll(Mx x);

  abstract class Stub implements Trans {
    public Trans gradient() {
      return null;
    }

//    public Mx transAll(Mx ds) {
//      Mx result = new VecBasedMx(ds.rows(), new ArrayVec(ds.rows() * ydim()));
//      for (int i = 0; i < ds.rows(); i++) {
//        VecTools.assign(result.col(i), trans(ds.row(i)));
//      }
//      return result;
//    }

    public Mx transAll(Mx ds) {
      Mx result = new VecBasedMx(ydim(), new ArrayVec(ds.rows() * ydim()));
      for (int i = 0; i < ds.rows(); i++) {
        VecTools.assign(result.row(i), trans(ds.row(i)));
      }
      return VecTools.transpose(result);
    }
  }
}
