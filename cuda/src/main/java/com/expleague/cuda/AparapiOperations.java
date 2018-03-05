package com.expleague.cuda;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.internal.kernel.KernelManager;
import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.math.vectors.impl.vectors.ArrayVec;

/**
 * Created by hrundelb on 20.11.17.
 */
public class AparapiOperations {

  public static float[] sum(final float[] left, final float[] right) {
    assert (left.length == right.length);
    final float[] result = new float[left.length];

    Kernel kernel = new Kernel() {
      @Override
      public void run() {
        int i = getGlobalId();
        result[i] = left[i] + right[i];
      }
    };

    Range range = Range.create(result.length);
    kernel.execute(range);
    return result;
  }

  public static void multiplyTo(float[] left, float[] right, float[] result, int size) {

    final int blockDim = 16;

    Kernel kernel = new Kernel() {

      @Override
      public void run() {
        int i = getGlobalId(0);
        int j = getGlobalId(1);

        float value = 0;
        for (int k = 0; k < size; k++) {
          value += left[i * size + k] * right[k * size + j];
        }
        result[i * size + j] = value;
      }
    };

    Device device = KernelManager.instance().bestDevice();
    System.out.println("Run on: " + device.getShortDescription());
    Range range = device.createRange2D(size, size, blockDim, blockDim);
    kernel.execute(range);
  }

  public static void multiplyTo(Mx left, Mx right, Mx result) {
    if(left.columns() != right.rows())
      throw new IllegalArgumentException();

    final int rows = left.rows();
    final int cols1 = left.columns();
    final int cols2 = right.columns();

    double[] arrayLeft = ((ArrayVec) ((VecBasedMx) left).vec).data.array;
    double[] arrayRight = ((ArrayVec) ((VecBasedMx) right).vec).data.array;
    double[] arrayResult = ((ArrayVec) ((VecBasedMx) result).vec).data.array;

    Kernel kernel = new Kernel() {
      @Override
      public void run() {
        int i = getGlobalId() / cols2;
        int j = getGlobalId() % cols2;

        double value = 0;
        for (int k = 0; k < cols1; k++) {
          value += arrayLeft[i * cols1 + k] * arrayRight[k * cols2 + j];
        }
        arrayResult[i * cols2 + j] = value;
      }
    };

    Device device = KernelManager.instance().bestDevice();
    System.out.println("Run on: " + device.getShortDescription());
    Range range = device.createRange(rows * cols2);
    kernel.execute(range);
  }



  public static void transpose(float[] left, float[] result, int size) {

    final int blockDim = 32;

    Kernel kernel = new Kernel() {

      @Local
      float[] temp = new float[blockDim * blockDim];

      @Override
      public void run() {
        int xIndex = getGlobalId(0);
        int yIndex = getGlobalId(1);

        if (xIndex < size && yIndex < size) {
          int inputIdx = xIndex * size + yIndex;
          temp[getLocalId(0) * blockDim + getLocalId(1)] = left[inputIdx];
        }

        localBarrier();

        if (xIndex < size && yIndex < size) {
          int outputIndex = yIndex * size + xIndex;
          result[outputIndex] = temp[getLocalId(0) * blockDim + getLocalId(1)];
        }
      }
    };

    Device device = KernelManager.instance().bestDevice();
    System.out.println("Run on: " + device.getShortDescription());
    Range range = device.createRange2D(size, size, blockDim, blockDim);
    kernel.execute(range);
  }


  public static void matrixExp(final float[] matrix, final float[] result, int rows, int passes) {

    Kernel kernel = new Kernel() {
      @Override
      public void run() {
        int i = getGlobalId();
        float sum = 0;
        for (int j = 0; j < rows - 1; j++) {
          float e = (float) Math.exp(matrix[i * (rows - 1) + j]);
          sum += e;
          result[i * rows + j] = e;
        }
        result[i * rows + rows - 1] = 1;
        sum += 1;
        for (int j = 0; j < rows; j++) {
          result[i * rows + j] = result[i * rows + j] / sum;
        }
      }
    };

    Device device = KernelManager.instance().bestDevice();
    System.out.println("Run on: " + device.getShortDescription());
    Range range = device.createRange(rows);
    kernel.execute(range, passes);
  }


  public static float[] vectorReduce(final float[] arguments) {

    final int blockSize = 32;
    final int n = arguments.length;
    final float[] results = new float[n];

    Kernel kernel = new Kernel() {

      @Local
      float[] sdata = new float[blockSize];

      @Override
      public void run() {
        final int tid = getLocalId();
        final int i = getGlobalId();

        if (i < n) {
          sdata[tid] = arguments[i];
        } else {
          //sdata[tid] = 1f;
        }

        for (int s = blockSize / 2; s > 0; s >>= 1) {
          if (tid < s) {
            sdata[tid] += sdata[tid + s];
          }
          localBarrier();
        }

        //if (tid == 0) {
          results[i] = sdata[tid];
        //}
      }
    };

    Device device = KernelManager.instance().bestDevice();
    System.out.println("Run on: " + device.getShortDescription());
    final int globalWidth = (int) Math.ceil((double) n / blockSize) * blockSize;
    Range range = device.createRange(globalWidth, blockSize);
    kernel.execute(range);
    return results;
  }


  public static void matrixExpReduce(final float[] matrix, final float[] result, int rows, int
      passes) {

    final int blockSize = rows;

    Kernel kernel = new Kernel() {
      @Local
      float[] sdata = new float[blockSize];

      @Local
      float[] res = new float[blockSize];

      @Override
      public void run() {
        final int i = getGlobalId();
        final int tid = getLocalId();
        final int blockId = getGroupId();

        if (tid < rows - 1) {
          sdata[tid] = exp(matrix[blockId * (rows - 1) + tid]);
          res[tid] = sdata[tid];
        } else {
          sdata[tid] = 1;
          res[tid] = 1;
        }

        localBarrier();

        for (int s = blockSize / 2; s > 0; s >>= 1) {
          if (tid < s) {
            sdata[tid] += sdata[tid + s];
          }
          localBarrier();
        }

        result[i] = res[tid] / sdata[0];
      }
    };

    Device device = KernelManager.instance().bestDevice();
    System.out.println("Run on: " + device.getShortDescription());
    Range range = device.createRange(rows * rows, blockSize);
    kernel.execute(range, passes);
  }

}
