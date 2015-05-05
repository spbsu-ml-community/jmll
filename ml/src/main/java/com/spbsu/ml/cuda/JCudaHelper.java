package com.spbsu.ml.cuda;

import org.jetbrains.annotations.NotNull;

import com.spbsu.commons.util.cache.Cache;
import com.spbsu.commons.util.cache.CacheStrategy;
import com.spbsu.commons.util.cache.impl.FixedSizeCache;
import com.spbsu.commons.io.StreamTools;
import jcuda.driver.*;
import com.spbsu.commons.system.RuntimeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.lang.reflect.Field;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

/**
 * jmll
 * ksen
 * 16.October.2014 at 11:35
 */
//todo(ksen): handle exit code
//todo(ksen): class/function name from Thread stack trace (+latency?)
public class JCudaHelper {

  private static final Logger LOG = LoggerFactory.getLogger(JCudaHelper.class);

  static {
    initJCuda();
  }

  private static final Cache<String, CUfunction> CACHE = new FixedSizeCache<>(100, CacheStrategy.Type.LRU);

  private static File LOCAL_PTX_DIRECTORY;

  private static CUcontext CONTEXT;

  public static void hook() {
    // outside init
  }

  public static void init() {
    JCudaDriver.setExceptionsEnabled(true);

    JCudaDriver.cuInit(0);
    final CUdevice device = new CUdevice();
    JCudaDriver.cuDeviceGet(device, 0);

    CONTEXT = new CUcontext();
    JCudaDriver.cuCtxCreate(CONTEXT, 0, device);
  }

  public static void destroy() {
    JCudaDriver.cuCtxDestroy(CONTEXT);
  }

  @NotNull
  public static CUfunction getFunction(final @NotNull String fileName, final @NotNull String functionName) {
    final String key = fileName + '.' + functionName;

    CUfunction function = CACHE.get(key);
    if (function != null) {
      return function;
    }
    final File ptxFile = new File(LOCAL_PTX_DIRECTORY, cuNameToPtx(fileName));

    final CUmodule module = new CUmodule();
    JCudaDriver.cuModuleLoad(module, ptxFile.getAbsolutePath());

    function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, module, functionName);
    CACHE.put(key, function);

    return function;
  }

  private static String cuNameToPtx(final String cuFileName) {
    final int extensionPoint = cuFileName.lastIndexOf('.');
    if (extensionPoint == -1) {
      throw new RuntimeException("Wrong extension " + cuFileName);
    }
    return cuFileName.substring(0, extensionPoint + 1) + "ptx";
  }

  // Static-magic stuff

  private static void initJCuda() {
    final ClassLoader classLoader = JCudaHelper.class.getClassLoader();
    try {
      final File tempDirectory = Files.createTempDirectory(JCudaConstants.JCUDA_TMP_DIRECTORY_NAME).toFile();
      setUsrPaths(tempDirectory.getAbsolutePath());

      LOG.info("Jcuda is working in: " + tempDirectory.getAbsolutePath());
      extractNativeLibraries(classLoader, tempDirectory);

      final File localCuDirectory = extractCuFiles(classLoader, tempDirectory);
      LOG.info("Local storage for *.cu files " + localCuDirectory.getAbsolutePath());

      LOCAL_PTX_DIRECTORY = compileCuFiles(localCuDirectory, tempDirectory);
      LOG.info("Local storage for *.ptx files " + LOCAL_PTX_DIRECTORY.getAbsolutePath());

      init();
    }
    catch (Exception e) {
      throw new RuntimeException(
          "Can't load Jcuda's native libraries. Are you sure what you have:\n" +
              "1. NVidia graphic card,\n" +
              "2. properly installed NVidia driver,\n" +
              "3. properly installed CUDA,\n" +
              "4. CUDA in environment variables (PATH, LD_LIBRARY_PATH),\n" +
              "5. Jcuda's dependencies with version = CUDA's version" +
              "on machine where you trying to run this code?",
          e
      );
    }
  }

  private static void setUsrPaths(final String path) {
    try {
      final Field usrPathsField = ClassLoader.class.getDeclaredField("usr_paths");
      usrPathsField.setAccessible(true);

      final String[] paths = (String[]) usrPathsField.get(null);
      final String[] newPaths = Arrays.copyOf(paths, paths.length + 1);
      newPaths[newPaths.length - 1] = path;

      usrPathsField.set(null, newPaths);
    }
    catch (NoSuchFieldException | IllegalAccessException e) {
      throw new RuntimeException("Something goes wrong while trying set usr_paths: " + path, e);
    }
  }

  private static void extractNativeLibraries(final ClassLoader classLoader, final File tempDirectory) {
    try {
      for (final String jcudaNativeLibName : JCudaConstants.JCUDA_NATIVE_LIBS_NAMES) {
        final URL resource = classLoader.getResource(jcudaNativeLibName);
        final File localReplica = new File(tempDirectory, "lib" + jcudaNativeLibName);
        try (
            final InputStream input = resource.openStream();
            final FileOutputStream output = new FileOutputStream(localReplica)
        ) {
          StreamTools.transferData(input, output);
        }
        LOG.info(jcudaNativeLibName + " extracted");
      }
    }
    catch (IOException e) {
      throw new RuntimeException("Something goes wrong while trying extract native libs: " + tempDirectory, e);
    }
  }

  private static File extractCuFiles(final ClassLoader classLoader, final File tempDirectory) {
    final File localCuDirectory = new File(tempDirectory, JCudaConstants.CU_CLASS_PATH);
    if (!localCuDirectory.mkdirs()) {
      throw new RuntimeException("Can't create local directory for *.cu " + localCuDirectory.getAbsolutePath());
    }

    final URL resource = classLoader.getResource(JCudaConstants.CU_CLASS_PATH.substring(1));
    if (resource == null) {
      throw new RuntimeException("Can't find *.cu directory in class path.");
    }

    final URI cuFilesUri;
    try {
      cuFilesUri = resource.toURI();
    }
    catch (URISyntaxException e) {
      throw new RuntimeException("Invalid url to *.cu directory in class path: " + resource);
    }

    final Path pathToCus = tryFindPathToCus(cuFilesUri);
    try {
      Files.walkFileTree(pathToCus, new SimpleFileVisitor<Path>() {
        @Override
        public FileVisitResult visitFile(final Path file, final BasicFileAttributes attributes) throws IOException {
          Files.copy(
              file,
              new File(localCuDirectory, file.getFileName().toString()).toPath(),
              StandardCopyOption.REPLACE_EXISTING
          );
          return FileVisitResult.CONTINUE;
        }
      });
    }
    catch (IOException e) {
      throw new RuntimeException(
          "Something goes wrong while trying transfer *.cu from " + pathToCus + " to " + localCuDirectory, e
      );
    }
    return localCuDirectory;
  }

  private static Path tryFindPathToCus(final URI cuFilesUri) {
    try {
      final Map<String, String> environment = new HashMap<>();
      environment.put("create", "true");

      final FileSystem fileSystem = FileSystems.newFileSystem(cuFilesUri, environment);
      return fileSystem.getPath(JCudaConstants.CU_CLASS_PATH);
    }
    catch (Exception e) {
      final FileSystem fileSystem = FileSystems.getDefault();
      return fileSystem.getPath(cuFilesUri.getPath());
    }
  }

  private static File compileCuFiles(final File localCuDirectory, final File tempDirectory) {
    final File localPtxDirectory = new File(tempDirectory, JCudaConstants.PTX_CLASS_PATH);
    if (!localPtxDirectory.mkdirs()) {
      throw new RuntimeException("Can't create local directory for *.ptx " + localPtxDirectory.getAbsolutePath());
    }

    for (final File cuFile : localCuDirectory.listFiles()) {
      compilePtx(cuFile, new File(localPtxDirectory, cuNameToPtx(cuFile.getName())));
    }
    return localPtxDirectory;
  }

  public static void compilePtx(final @NotNull File cuFile, final @NotNull File ptxFile) {
    final String command = new StringBuilder()
        .append("nvcc ")
        .append("-m ").append(RuntimeUtils.getArchDataModel()).append(' ')
        .append("-ptx ").append(cuFile.getAbsolutePath()).append(' ')
        .append("-o ").append(ptxFile.getAbsolutePath())
        .toString()
    ;

    final int exitCode = execNvcc(command);
    if (exitCode != 0) {
      throw new RuntimeException("Could not create *.ptx file: " + ptxFile.getAbsolutePath());
    }
  }

  private static int execNvcc(final String command) {
    int exitCode;
    String stdErr = "";
    String stdOut = "";

    try {
      final Process process = Runtime.getRuntime().exec(command);

      stdErr = streamToString(process.getErrorStream());
      stdOut = streamToString(process.getInputStream());

      exitCode = process.waitFor();
    }
    catch (IOException | InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException(
          "Interrupted while waiting for nvcc output.\nSTDOUT:\n" + stdOut + "\nSTDERR\n" + stdErr, e
      );
    }
    return exitCode;
  }

  private static String streamToString(final InputStream inputStream) throws IOException {
    final StringBuilder builder = new StringBuilder();
    try (final LineNumberReader reader = new LineNumberReader(new InputStreamReader(inputStream))) {
      final char[] buffer = new char[8192];

      int read;
      while ((read = reader.read(buffer)) != -1) {
        builder.append(buffer, 0, read);
      }
    }
    return builder.toString();
  }

}
