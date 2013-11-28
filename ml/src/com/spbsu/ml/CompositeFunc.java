package com.spbsu.ml;

import com.spbsu.commons.func.Evaluator;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.ml.func.VecTransform;
import org.jetbrains.annotations.Nullable;

/**
 * User: solar
 * Date: 21.12.2010
 * Time: 22:07:07
 */
public class CompositeFunc<F extends Func, G extends VecFunc> extends FuncStub implements Func {
  public final F f;
  public final G g;

  public CompositeFunc(F f, G g) {
    this.f = f;
    this.g = g;
  }

  @Override
  public int xdim() {
    return g.direction(ArrayTools.max(g.directions(), new Evaluator<Func>() {
      @Override
      public double value(Func func) {
        return func.xdim();
      }
    })).xdim();
  }

  @Nullable
  @Override
  public VecFunc gradient() {
    return new VecTransform() {
      @Override
      public Vec vvalue(Vec x) {
        return VecTools.multiply(VecTools.transpose(g.gradient(x)), f.gradient().vvalue(g.vvalue(x)));
      }

      @Override
      public int xdim() {
        return g.xdim();
      }
    };
  }

  @Override
  public double value(Vec x) {
    return f.value(g.vvalue(x));
  }
}
