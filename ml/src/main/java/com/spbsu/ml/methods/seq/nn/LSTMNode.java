package com.spbsu.ml.methods.seq.nn;

import com.spbsu.commons.math.vectors.SingleValueVec;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.func.generic.Sigmoid;
import com.spbsu.ml.func.generic.Tanh;

public class LSTMNode implements NetworkNode {
  private final Vec params;
  private final Vec wForget;
  private final Vec bForget;
  private final Vec wInput;
  private final Vec bInput;
  private final Vec wCandidate;
  private final Vec bCandidate;
  private final Vec wOutput;
  private final Vec bOutput;

  private final int forgetStart;
  private final int inputStart;
  private final int candidateStart;
  private final int outputStart;

  /**
   *
   * @param inputDim dim of input (not including previous cell value)
   * @param random
   */
  public LSTMNode(int inputDim, FastRandom random) {
    inputDim += 1; //for previous cell value

    params = new ArrayVec(4 * (inputDim + 1));
    for (int i = 0; i < params.dim(); i++) {
      if (i % (inputDim + 1) != 0) {
        params.set(i, random.nextGaussian() / inputDim);
      }
    }
    forgetStart = 0;
    wForget = params.sub(forgetStart, inputDim);
    bForget = params.sub(forgetStart + inputDim, 1);

    inputStart = inputDim + 1;
    wInput = params.sub(inputStart, inputDim);
    bInput = params.sub(inputStart + inputDim, 1);

    candidateStart = 2 * (inputDim + 1);
    wCandidate = params.sub(candidateStart, inputDim);
    bCandidate = params.sub(candidateStart + inputDim, 1);

    outputStart = 3 * (inputDim + 1);
    wOutput = params.sub(outputStart, inputDim);
    bOutput = params.sub(outputStart + inputDim, 1);
  }

  @Override
  public Vec params() {
    return params;
  }

  /**
   *
   * @param x
   * @return gradient by input - vector of (grad by prev output, grad by prev cell value), gradient by params
   */
  //Fixme: not calculating gradient by input, only by previous node values
  @Override
  public NetworkNode.NodeGrad grad(Vec x, Vec nodeOutputGrad) {
    final FunctionValues values = new FunctionValues(x);
    final Vec gradByParams = new ArrayVec(params.dim());
    final Vec gradByInput = new ArrayVec(2);
    final Tanh tanh = new Tanh();

    final Vec xAndOutput = x.sub(0, x.dim() - 1);
    final double prevCellValue = x.get(x.dim() - 1);

    final double outputGrad = tanh.value(new SingleValueVec(values.cellValue)); //d cellOutput / d output

    adjustGrad(gradByParams, outputStart, xAndOutput, outputGrad * values.outputGrad * nodeOutputGrad.get(0));


    final double cellValueGrad = values.output * tanh.gradient(new SingleValueVec(values.cellValue)).get(0);

    final double forgetGrad = cellValueGrad * prevCellValue; //d cellOutput / d forget
    adjustGrad(gradByParams, forgetStart, xAndOutput, forgetGrad * values.forgetGrad * nodeOutputGrad.get(0));
    adjustGrad(gradByParams, forgetStart, xAndOutput, prevCellValue * values.forgetGrad * nodeOutputGrad.get(1));


    final double inputGrad = cellValueGrad * values.candidate; //d cellOutput / d input
    final double candidateGrad = cellValueGrad * values.input;//d cellOutput / d candidate

    adjustGrad(gradByParams, inputStart, xAndOutput, inputGrad * values.inputGrad * nodeOutputGrad.get(0));
    adjustGrad(gradByParams, inputStart, xAndOutput, values.candidate * values.inputGrad * nodeOutputGrad.get(1));

    adjustGrad(gradByParams, candidateStart, xAndOutput, candidateGrad * values.candidateGrad * nodeOutputGrad.get(0));
    adjustGrad(gradByParams, candidateStart, xAndOutput, values.input * values.candidateGrad * nodeOutputGrad.get(1));


    gradByInput.adjust(0, outputGrad * values.outputGrad * wOutput.get(xAndOutput.dim() - 1) * nodeOutputGrad.get(0));
    gradByInput.adjust(0, forgetGrad * values.forgetGrad * wForget.get(xAndOutput.dim() - 1) * nodeOutputGrad.get(0));
    gradByInput.adjust(0, inputGrad * values.inputGrad * wInput.get(xAndOutput.dim() - 1) * nodeOutputGrad.get(0));
    gradByInput.adjust(0, candidateGrad * values.candidateGrad * wCandidate.get(xAndOutput.dim() - 1) * nodeOutputGrad.get(0));

    gradByInput.adjust(0, prevCellValue * values.forgetGrad * wForget.get(xAndOutput.dim() - 1) * nodeOutputGrad.get(1));
    gradByInput.adjust(0, values.candidate * values.inputGrad * wInput.get(xAndOutput.dim() - 1) * nodeOutputGrad.get(1));
    gradByInput.adjust(0, values.input * values.candidateGrad * wCandidate.get(xAndOutput.dim() - 1) * nodeOutputGrad.get(1));

    gradByInput.set(1, values.forget * nodeOutputGrad.get(1) + cellValueGrad * values.forget * nodeOutputGrad.get(0));

    return new NodeGrad(gradByParams, gradByInput);
  }

  private void adjustGrad(Vec grad, int start, Vec vec, double scale) {
    VecTools.incscale(grad.sub(start, vec.dim()), vec, scale);
    grad.adjust(start + vec.dim(), scale);
  }

  /**
   *
   * @param x vector of (input vector, previous output value, previous cell value)
   * @return vector of (cell output, cell value)
   */
  @Override
  public Vec value(Vec x) {
    final FunctionValues values = new FunctionValues(x);
    return new ArrayVec(values.cellOutput, values.cellValue);
  }

  private class FunctionValues {
    double forget;
    double forgetGrad;
    double input;
    double inputGrad;
    double candidate;
    double candidateGrad;
    double cellValue;
    double output;
    double outputGrad;
    double cellOutput;

    FunctionValues(Vec x) {
      final Sigmoid sigmoid = new Sigmoid();
      final Tanh tanh = new Tanh();
      final Vec xAndOutput = x.sub(0, x.dim() - 1);
      final double prevCellValue = x.get(x.dim() - 1);

      final Vec forgetX = new SingleValueVec(VecTools.multiply(wForget, xAndOutput) + bForget.get(0));
      forget = sigmoid.value(forgetX);
      forgetGrad = sigmoid.gradient(forgetX).get(0);

      final Vec inputX = new SingleValueVec(VecTools.multiply(wInput, xAndOutput) + bInput.get(0));
      input = sigmoid.value(inputX);
      inputGrad = sigmoid.gradient(inputX).get(0);

      final Vec candidateX = new SingleValueVec(VecTools.multiply(wCandidate, xAndOutput) + bCandidate.get(0));
      candidate = tanh.value(candidateX);
      candidateGrad = tanh.gradient(candidateX).get(0);

      cellValue = forget * prevCellValue + input * candidate;

      final Vec outputX = new SingleValueVec(VecTools.multiply(wOutput, xAndOutput) + bOutput.get(0));
      output = sigmoid.value(outputX);
      outputGrad = sigmoid.gradient(outputX).get(0);

      cellOutput = output * tanh.value(new SingleValueVec(cellValue));
    }
  }
}
