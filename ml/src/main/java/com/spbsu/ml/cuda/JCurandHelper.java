package com.spbsu.ml.cuda;

import org.jetbrains.annotations.NotNull;

import jcuda.driver.CUdeviceptr;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class JCurandHelper {

  static {
    JCudaHelper.hook();
  }

  public static curandGenerator createDefault() {
    return createGenerator(curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
  }

  public static curandGenerator createGenerator(final int type) {
    final curandGenerator generator = new curandGenerator();

    JCurand.curandCreateGenerator(generator, type);
    JCurand.curandSetPseudoRandomGeneratorSeed(generator, System.nanoTime());

    return generator;
  }

  public static void destroyGenerator(final @NotNull curandGenerator generator) {
    JCurand.curandDestroyGenerator(generator);
  }

  public static CUdeviceptr generateUniform(final int length, final @NotNull curandGenerator generator) {
    final CUdeviceptr devicePointer = JCudaMemory.allocFloat(length);

    return generateUniform(devicePointer, length, generator);
  }

  public static CUdeviceptr generateUniform(
      final @NotNull CUdeviceptr devicePointer,
      final long length,
      final @NotNull curandGenerator generator
  ) {
    JCurand.curandGenerateUniform(generator, devicePointer, length);

    return devicePointer;
  }

  public static float[] generateUniformHost(final int size, final @NotNull curandGenerator generator) {
    final CUdeviceptr devicePointer = JCudaMemory.allocFloat(size);

    JCurand.curandGenerateUniform(generator, devicePointer, size);

    return JCudaMemory.copyFloatsDestr(devicePointer, size);
  }

}
