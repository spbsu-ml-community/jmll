package com.spbsu.exp.dl.dnn;

import com.spbsu.commons.math.vectors.MxTools;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.exp.dl.dnn.rectifiers.Rectifier;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;

/**
 * jmll
 *
 * @author ksenon
 */
public class Layer {

  public Mx input;
  public Mx output;
  public Mx weights;
  public Mx difference;
  public Mx activations;
  public Mx dropoutMask;
  public Rectifier rectifier;

  public double dropoutFraction;
  public boolean isTrain;

  public boolean debug;

  public void forward() {
    output = MxTools.multiply(input, MxTools.transpose(weights));
    rectifier.value(output, output);

    if (dropoutFraction > 0) {
      if (isTrain) {
        dropoutMask = getDropoutMask();

        for (int i = 0; i < output.dim(); i++) {
          output.set(i, output.get(i) * dropoutMask.get(i));
        }
      }
      else {
        for (int i = 0; i < output.dim(); i++) {
          output.set(i, output.get(i) * (1 - dropoutFraction));
        }
      }
    }
    activations = VecTools.copy(output);
  }

  private Mx getDropoutMask() {
    final Mx dropoutMask = new VecBasedMx(output.rows(), output.columns());
    for (int i = 0; i < dropoutMask.dim(); i++) {
      dropoutMask.set(i, Math.random() > dropoutFraction ? 1 : 0);
    }
    return dropoutMask;
  }

  public void backward(final Mx perv) {
    difference = MxTools.multiply(MxTools.transpose(output), perv);

    input = MxTools.multiply(output, weights);

    rectifier.grad(perv, perv);
    for (int i = 0; i < input.dim(); i++) {
      input.set(i, input.get(i) * perv.get(i));
      if (dropoutFraction > 0) {
        input.set(i, input.get(i) * dropoutMask.get(i));
      }
    }
  }

}
