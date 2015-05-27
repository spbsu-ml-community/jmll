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

  public double bias;
  public double bias_b;
  public double dropoutFraction;
  public boolean isTrain;

  public boolean debug;

  public void forward() {
    if (bias != 0) {
      activations = leftExtend(input);
    }
    else {
      activations = VecTools.copy(input);
    }

    output = MxTools.multiply(activations, MxTools.transpose(weights));
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
  }

  private Mx leftExtend(final Mx original) {
    final VecBasedMx extended = new VecBasedMx(original.rows(), original.columns() + 1);

    for (int i = 0; i < original.rows(); i++) {
      extended.set(i, 0, bias);
    }

    for (int i = 0; i < original.rows(); i++) {
      for (int j = 1; j < original.columns() + 1; j++) {
        extended.set(i, j, original.get(i, j - 1));
      }
    }
    return extended;
  }

  private Mx getDropoutMask() {
    final Mx dropoutMask = new VecBasedMx(output.rows(), output.columns());
    for (int i = 0; i < dropoutMask.dim(); i++) {
      dropoutMask.set(i, Math.random() > dropoutFraction ? 1 : 0);
    }
    return dropoutMask;
  }

  public void backward() {
    Mx cnc = null;
    if (bias_b != 0) {
      cnc = leftContract(output);
    }
    else {
      cnc = VecTools.copy(output);
    }

    difference = MxTools.multiply(MxTools.transpose(cnc), activations);
    for (int i = 0; i < difference.dim(); i++) {
      difference.set(i, difference.get(i) / activations.rows());
    }

    input = MxTools.multiply(cnc, weights);

    rectifier.grad(activations, activations);
    for (int i = 0; i < input.dim(); i++) {
      input.set(i, input.get(i) * activations.get(i));
      if (dropoutFraction > 0) {
        input.set(i, input.get(i) * dropoutMask.get(i));
      }
    }
  }

  private Mx leftContract(final Mx original) {
    final VecBasedMx contracted = new VecBasedMx(original.rows(), original.columns() - 1);

    for (int i = 0; i < contracted.rows(); i++) {
      for (int j = 0; j < contracted.columns(); j++) {
        contracted.set(i, j, original.get(i, j + 1));
      }
    }
    return contracted;
  }

}
