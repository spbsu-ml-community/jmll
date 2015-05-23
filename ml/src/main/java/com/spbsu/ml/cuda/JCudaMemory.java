package com.spbsu.ml.cuda;

import org.jetbrains.annotations.NotNull;

import jcuda.driver.CUresult;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class JCudaMemory {

  static {
    JCudaHelper.hook();
  }


  // Allocation

  public static CUdeviceptr allocInt(final long length) {
    return alloc(length, Sizeof.INT);
  }

  public static CUdeviceptr allocFloat(final long length) {
    return alloc(length, Sizeof.FLOAT);
  }

  public static CUdeviceptr allocLong(final long length) {
    return alloc(length, Sizeof.LONG);
  }

  public static CUdeviceptr allocDouble(final long length) {
    return alloc(length, Sizeof.DOUBLE);
  }

  public static CUdeviceptr alloc(final long length, final int size) {
    final CUdeviceptr devicePointer = new CUdeviceptr();

    final int response = JCudaDriver.cuMemAlloc(devicePointer, length * size);
    checkResponse(
        response,
        "Allocation failed. Length = {0}, size = {1}.",
        length, size
    );

    return devicePointer;
  }


  // Duplication (Host -> Device)

  public static CUdeviceptr copy(final float[] data, final @NotNull CUdeviceptr devicePointer) {
    return copy(Pointer.to(data), devicePointer, data.length * Sizeof.FLOAT);
  }

  public static CUdeviceptr copy(
      final @NotNull Pointer hostPointer,
      final @NotNull CUdeviceptr devicePointer,
      final long bytes
  ) {
    final int response = JCudaDriver.cuMemcpyHtoD(devicePointer, hostPointer, bytes);
    checkResponse(
        response,
        "Memory duplication failed(H -> D). From {0} to {1} ({2} bytes).",
        hostPointer, devicePointer, bytes
    );

    return devicePointer;
  }


  // Duplication (Device -> Host)

  public static float[] copy(final @NotNull CUdeviceptr devicePointer, final int length) {
    return copy(devicePointer, new float[length]);
  }

  public static float[] copy(final @NotNull CUdeviceptr devicePointer, final float[] hostData) {
    final int length = hostData.length;
    final Pointer hostPointer = Pointer.to(hostData);

    final int response = JCudaDriver.cuMemcpyDtoH(hostPointer, devicePointer, length * Sizeof.FLOAT);
    checkResponse(
        response,
        "Memory duplication failed(D -> H). From {0} to {1} ({2} floats).",
        hostPointer, devicePointer, length
    );

    return hostData;
  }


  // Duplication (Device -> Device)

  public static void insertFloats(
      final @NotNull CUdeviceptr source,
      final @NotNull CUdeviceptr destination,
      final long destinationLength
  ) {
    final int response = JCudaDriver.cuMemcpyDtoD(destination, source, destinationLength * Sizeof.FLOAT);
    checkResponse(
        response,
        "Memory transfer failed(D -> D). From {0} to {1} ({3} floats).",
        source, destination, destinationLength
    );
  }

  public static void insertFloats(
      final @NotNull CUdeviceptr source,
      final long sourceOffset,
      final @NotNull CUdeviceptr destination,
      final long destinationOffset,
      final long destinationLength
  ) {
    final CUdeviceptr shiftedSource = source.withByteOffset(sourceOffset * Sizeof.FLOAT);
    final CUdeviceptr shiftedDestination = destination.withByteOffset(destinationOffset * Sizeof.FLOAT);
    final int response = JCudaDriver.cuMemcpyDtoD(shiftedDestination, shiftedSource, destinationLength * Sizeof.FLOAT);
    checkResponse(
        response,
        "Memory transfer failed(D -> D). " +
            "From {0}(shifted {1} by {2} floats) to {3}(shifted {4} by {5} floats) {6} floats.",
        source, shiftedSource, sourceOffset, destination, shiftedDestination, destinationOffset, destinationLength
    );
  }


  // Allocation and duplication (Host -> Device)

  public static CUdeviceptr alloCopy(final float[] data) {
    final long length = data.length;

    final CUdeviceptr devicePointer = allocFloat(length);
    copy(data, devicePointer);

    return devicePointer;
  }

  public static CUdeviceptr alloCopy(final double[] data) {
    final int length = data.length;

    final CUdeviceptr devicePointer = allocLong(length);
    JCudaDriver.cuMemcpyHtoD(devicePointer, Pointer.to(data), length * Sizeof.DOUBLE);

    return devicePointer;
  }


  // Purification

  public static void destroy(final @NotNull CUdeviceptr devicePointer) {
    final int response = JCudaDriver.cuMemFree(devicePointer);
    checkResponse(
        response,
        "Device memory purification failed ({0}).",
        devicePointer
    );
  }


  // Duplication (Device -> Host) and purification

  public static float[] copyFloatsDestr(final @NotNull CUdeviceptr devicePointer, final int length) {
    return copyFloatsDestr(devicePointer, new float[length]);
  }

  public static float[] copyFloatsDestr(final @NotNull CUdeviceptr devicePointer, final float[] hostData) {
    JCudaDriver.cuMemcpyDtoH(Pointer.to(hostData), devicePointer, hostData.length * Sizeof.FLOAT);
    JCudaDriver.cuMemFree(devicePointer);

    return hostData;
  }

  public static double[] copyDestr(final @NotNull CUdeviceptr devicePointer, final int length) {
    return copyDestr(devicePointer, new double[length]);
  }

  public static double[] copyDestr(final @NotNull CUdeviceptr devicePointer, final double[] hostData) {
    JCudaDriver.cuMemcpyDtoH(Pointer.to(hostData), devicePointer, hostData.length * Sizeof.DOUBLE);
    JCudaDriver.cuMemFree(devicePointer);

    return hostData;
  }

  private static void checkResponse(final int response, final String format, final Object ... arguments) {
    if (response != CUresult.CUDA_SUCCESS) {
      throw new CudaException(response, format, arguments);
    }
  }

}
