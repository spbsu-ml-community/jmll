package com.spbsu.ml.cuda.data;

import org.jetbrains.annotations.NotNull;

import jcuda.driver.CUdeviceptr;

/**
 * Project jmll
 *
 * @author Ksen
 */
public interface ArrayBased<BASE> {

  @NotNull CUdeviceptr reproduce();

  @NotNull ArrayBased<BASE> set(final @NotNull BASE hostArray);

  @NotNull ArrayBased<BASE> reset(final @NotNull BASE hostArray);

  @NotNull BASE get();

  void setPointer(final @NotNull CUdeviceptr devicePointer);

  @NotNull CUdeviceptr getPointer();

  long length();

  void destroy();

}
