package com.expleague.ml.testUtils;

import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.data.tools.DataTools;

import java.io.*;
import java.net.URL;
import java.util.zip.GZIPInputStream;

/**
 * User: qdeee
 * Date: 23.07.14
 */
public final class TestResourceLoader {
  private TestResourceLoader() {}

  public static InputStream loadResourceAsStream(final String localPath) throws IOException{
    final InputStream resource = TestResourceLoader.class.getClassLoader().getResourceAsStream
        ("com/expleague/ml/" + localPath);
    if (resource == null) {
      throw new IOException("Resource \"" + localPath + "\" not found");
    }
    return resource;
  }

  public static String getFullPath(final String localPath) throws IOException {
    final URL resource = TestResourceLoader.class.getClassLoader().getResource
        ("com/expleague/ml/" + localPath);
    if (resource == null) {
      throw new IOException("Resource \"" + localPath + "\" not found");
    }
    return resource.getPath();
  }

  public static Pool<?> loadPool(final String localPath) throws IOException {
    final InputStream stream = loadResourceAsStream(localPath);
    final InputStreamReader reader = localPath.endsWith(".gz") ? new InputStreamReader(new GZIPInputStream(stream))
                                                               : new InputStreamReader(stream);
    return DataTools.loadFromFeaturesTxt(localPath, reader);
  }

}
