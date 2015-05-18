package com.spbsu.ml.cuda;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import jcuda.driver.CUresult;

import static java.text.MessageFormat.*;

/**
 * Project jmll
 *
 * @author Ksen
 */
public class CudaException extends RuntimeException {

  private static final String HEADER = "Cuda returned {0}({1}). ";

  public CudaException(final int errorCode, final @NotNull String format, final Object ... arguments) {
    this(format(HEADER, CUresult.stringFor(errorCode), errorCode) + format, arguments);
  }

  public CudaException(final @NotNull String format, final Object ... arguments) {
    this(format(format, arguments));
  }

  public CudaException(final Throwable throwable, final @NotNull String format, final Object ... arguments) {
    this(throwable, format(format, arguments));
  }

  public CudaException(final Throwable throwable, final @Nullable String message) {
    super(message, throwable);
  }

  public CudaException(final @Nullable String message) {
    super(message);
  }

}
