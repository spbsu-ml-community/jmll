package com.spbsu.ml.cuda;

import org.jetbrains.annotations.NotNull;
import com.spbsu.commons.util.cache.Cache;
import com.spbsu.commons.util.cache.CacheStrategy;
import com.spbsu.commons.util.cache.impl.FixedSizeCache;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.util.logging.Logger;
import jcuda.driver.*;
import com.spbsu.commons.system.RuntimeUtils;
import jcuda.Pointer;
import jcuda.Sizeof;

import java.io.*;
import java.lang.reflect.Field;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;

import static jcuda.driver.JCudaDriver.*;

/**
 * jmll
 * ksen
 * 16.October.2014 at 11:35
 */
public class JcudaHelper {

  private static final Logger LOG = Logger.create(JcudaHelper.class);

  private static File LOCAL_PTX_DIRECTORY;
  private static final Cache<String, CUfunction> CACHE = new FixedSizeCache<>(100, CacheStrategy.Type.LRU);

  private static CUcontext CONTEXT;

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
      LOG.warn("Wrong extension " + cuFileName);
    }
    return cuFileName.substring(0, extensionPoint + 1) + "ptx";
  }

  public static void hook() {}

  static {
    final ClassLoader classLoader = JcudaHelper.class.getClassLoader();
    try {
      final File tempDirectory = Files.createTempDirectory(JcudaConstants.JCUDA_TMP_DIRECTORY_NAME).toFile();
      tempDirectory.deleteOnExit();
      setUsrPaths(tempDirectory.getAbsolutePath());

      LOG.info("Jcuda is working in the " + tempDirectory.getAbsolutePath());
      extractNativeLibraries(classLoader, tempDirectory);

      final File localCuDirectory = extractCuFiles(classLoader, tempDirectory);
      LOG.info("Local storage for a *.cu files " + localCuDirectory.getAbsolutePath());

      LOCAL_PTX_DIRECTORY = compileCuFiles(localCuDirectory, tempDirectory);
      LOG.info("Local storage for a *.ptx files " + LOCAL_PTX_DIRECTORY.getAbsolutePath());

      init();
    }
    catch (Exception e) {
      LOG.error(
          "Can't load Jcuda's native libraries. Are you sure what you have:\n" +
              "1. NVidia graphic card,\n" +
              "2. properly installed NVidia driver,\n" +
              "3. properly installed CUDA,\n" +
              "4. CUDA in a environment variables (LD_LIBRARY_PATH),\n" +
              "5. Jcuda's dependencies with version = CUDA's version" +
              "on machine where you trying to run this code?"
      );
    }
  }

  private static void setUsrPaths(final String path) throws NoSuchFieldException, IllegalAccessException {
    final Field usrPathsField = ClassLoader.class.getDeclaredField("usr_paths");
    usrPathsField.setAccessible(true);

    final String[] paths = (String[]) usrPathsField.get(null);
    final String[] newPaths = Arrays.copyOf(paths, paths.length + 1);
    newPaths[newPaths.length - 1] = path;

    usrPathsField.set(null, newPaths);
  }

  private static void extractNativeLibraries(final ClassLoader classLoader, final File tempDirectory)
      throws IOException
  {
    for (final String jcudaNativeLibName : JcudaConstants.JCUDA_NATIVE_LIBS_NAMES) {
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

  private static File extractCuFiles(final ClassLoader classLoader, final File tempDirectory)
      throws URISyntaxException, IOException
  {
    final File localCuDirectory = new File(tempDirectory, JcudaConstants.CU_CLASS_PATH);
    if (!localCuDirectory.mkdirs()) {
      LOG.error("Can't create local directory for a *.cu " + localCuDirectory.getAbsolutePath());
      throw new RuntimeException();
    }
    final URL resource = classLoader.getResource(JcudaConstants.CU_CLASS_PATH.substring(1));
    if (resource == null) {
      LOG.error("Can't find *.cu directory.");
      throw new RuntimeException();
    }
    final URI cuFilesUri = resource.toURI();
    final Map<String, String> environment = new HashMap<>();
    environment.put("create", "true");

    try (final FileSystem jarFileSystem = FileSystems.newFileSystem(cuFilesUri, environment)) {
      final Path path = jarFileSystem.getPath(JcudaConstants.CU_CLASS_PATH);
      Files.walkFileTree(path, new SimpleFileVisitor<Path>(){
        @Override
        public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
          Files.copy(
              file,
              new File(localCuDirectory, file.getFileName().toString()).toPath(),
              StandardCopyOption.REPLACE_EXISTING
          );
          return FileVisitResult.CONTINUE;
        }
      });
    }
    return localCuDirectory;
  }

  private static File compileCuFiles(final File localCuDirectory, final File tempDirectory) {
    final File localPtxDirectory = new File(tempDirectory, JcudaConstants.PTX_CLASS_PATH);
    localPtxDirectory.mkdirs();
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
    final int exitCode;
    final String stdErr;
    final String stdOut;
    try {
      final Process process = Runtime.getRuntime().exec(command);

      stdErr = streamToString(process.getErrorStream());
      stdOut = streamToString(process.getInputStream());
      exitCode = process.waitFor();
    }
    catch (Exception e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("Interrupted while waiting for nvcc output", e);
    }
    if (exitCode != 0) {
      LOG.error("nvcc ended with exit code " + exitCode + "\nstderr: " + stdErr + "\nstdout: " + stdOut);
      throw new RuntimeException("Could not create *.ptx file: " + ptxFile.getAbsolutePath());
    }
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
