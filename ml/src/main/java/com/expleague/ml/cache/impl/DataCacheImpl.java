package com.expleague.ml.cache.impl;

import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import com.expleague.ml.cache.DataCache;
import com.expleague.ml.cache.DataCacheConfig;
import com.expleague.ml.cache.DataCacheItem;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.util.StdDateFormat;
import de.schlichtherle.truezip.file.TArchiveDetector;
import de.schlichtherle.truezip.file.TConfig;
import de.schlichtherle.truezip.fs.archive.zip.ZipDriver;
import de.schlichtherle.truezip.nio.file.TPath;
import de.schlichtherle.truezip.socket.sl.IOPoolLocator;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.OutputStream;
import java.io.Writer;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.*;
import java.util.stream.Stream;

public class DataCacheImpl<D, Conf extends DataCacheConfig> implements DataCache<D, Conf> {
  private static final Logger log = LoggerFactory.getLogger(DataCacheImpl.class);

  private final Path path;
  private final Class<Conf> configClass;
  private final List<Class<? extends DataCacheItem.Stub>> partClasses;
  private List<DataCacheItem> parts = new ArrayList<>();

  private final D dao;
  private Conf config;

  private LoggerContext loggerCtxt = new LoggerContext();
  private ArchiveAppender archiveAppender = new ArchiveAppender();

  private final ObjectMapper mapper;

  @SafeVarargs
  public DataCacheImpl(Path path, D dao, Class<Conf> configClass, Class<? extends DataCacheItem.Stub<?, D, ? super Conf>>... parts) {
    this.path = path;
    this.dao = dao;
    this.configClass = configClass;
    partClasses = new ArrayList<>(Arrays.asList(parts));
    partClasses.add(DataCacheLog.class);

    {
      archiveAppender.setContext(loggerCtxt);
    }

    for (Class<? extends DataCacheItem> partClass : parts) {
      try {
        //noinspection JavaReflectionMemberAccess
        partClass.getDeclaredConstructor();
      } catch (NoSuchMethodException e) {
        throw new IllegalArgumentException(partClass.getName() + " has no empty constructor.");
      }
    }
    {
      mapper = new ObjectMapper();
      mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }
  }

  public Class<? extends DataCacheItem.Stub>[] available() {
    //noinspection unchecked
    return partClasses.toArray(new Class[partClasses.size()]);
  }

  @Override
  public boolean contains(Class<? extends DataCacheItem> part) {
    //noinspection unchecked
    DataCacheItem<?, D, Conf> item = item(part);
    return Files.exists(item.getPath());
  }

  private Stack<Set<String>> dependencyPool = new Stack<>();
  @Override
  public synchronized void update(Class<? extends DataCacheItem> itemClass) {
    //noinspection unchecked
    DataCacheItem<?, D, Conf> item = item(itemClass);
    TPath entry = item.getPath();
    Path tempFile;
    if (dependencyPool.empty()) {
      archiveAppender.start();
    }
    dependencyPool.push(new HashSet<>());
    item.getLogger().info("Update start");
    String suffix = "." + (entry.getFileName() != null ? entry.getFileName().toString() : "no-file");
    try (OutputStream out = Files.newOutputStream(tempFile = Files.createTempFile("data-cache-item-", suffix))) {
      Instant updateStart = Instant.now();
      item.update(out);
      out.close();

      if (Files.exists(entry)) {
        if (Files.getLastModifiedTime(entry).toInstant().isBefore(updateStart)) { // in case of chain call for update from wrapper components like Map
          Files.move(tempFile, entry, StandardCopyOption.REPLACE_EXISTING);
        }
      } else {
        Files.move(tempFile, entry);
      }
      item.getLogger().info("Build successful");
      String[] dependsOn = dependencyPool.pop().toArray(new String[0]);
      getConfig().updateDependencies(itemClass, dependsOn);
      getConfig().notifyUpdated(itemClass.getName());
    } catch (IOException e) {
      item.getLogger().info("Build failed: " + e.getMessage());
      throw new RuntimeException(e);
    } finally {
      if (dependencyPool.empty()) {
        try (Stream<CharSequence> log = this.archiveAppender.flush()) {
          TPath logPath = new TPath(getPath() + "/log.txt");
          try (Writer writer = Files.newBufferedWriter(logPath, StandardCharsets.UTF_8, Files.exists(logPath) ? StandardOpenOption.APPEND : StandardOpenOption.CREATE_NEW)) {
            log.forEach(msg -> {
              try {
                writer.append(msg).append('\n');
              } catch (IOException e) {
                throw new RuntimeException(e);
              }
            });
          }
        } catch (IOException | RuntimeException ignored) {
          log.warn("Unable to update log.txt entry");
        }
        this.archiveAppender.stop();
      }
    }
  }

  @Override
  public <T, P extends DataCacheItem<T, ? super D, ?>> T get(Class<P> partClass) {
    P part = item(partClass);
    if (!dependencyPool.empty()) {
      dependencyPool.peek().add(partClass.getName());
    }
    TPath entry = part.getPath();
    Conf config = getConfig();

    final List<String> updates = new ArrayList<>();
    if (!Files.exists(entry) || !config.isUpToDate(partClass, updates)) { // no entry building from scratch
      if (!updates.isEmpty()) {
        log.info("Updating " + partClass + " because it is older (" + config.updateTime(partClass) + ") then: ");
        updates.forEach(p -> {
          log.info("\t" + p + ": " + config.updateTime(p));
        });
      }
      update(partClass);
    }
    try {
      return part.read(Files.newInputStream(entry));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public Path getPath() {
    return path.toAbsolutePath();
  }

  public D getDataManager() {
    return dao;
  }

  @Override
  public Conf getConfig() {
    if (config == null) {
      try {
        TPath configPath = getConfigPath();
        if (Files.exists(configPath)) {
          config = mapper.readValue(Files.newInputStream(configPath), configClass);
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      if (config == null) {
        try {
          config = configClass.newInstance();
          flushConfig();
        } catch (InstantiationException | IllegalAccessException e) {
          throw new RuntimeException(e);
        }
      }
      Class<DataCacheConfig> configClass = DataCacheConfig.class;
      try {
        {
          Field id = configClass.getDeclaredField("id");
          id.setAccessible(true);
          id.set(config, UUID.randomUUID().toString());
        }
        {
          Field owner = configClass.getDeclaredField("owner");
          owner.setAccessible(true);
          owner.set(config, this);
        }
      } catch (NoSuchFieldException | IllegalAccessException e) {
        throw new RuntimeException(e);
      }

    }
    return config;
  }

  private <R extends DataCacheItem<?, ? super D, ?>> R item(Class<R> partClass) {
    //noinspection unchecked
    Class<? extends R> realPartClass = (Class<? extends R>) Stream.of(available())
        .filter(partClass::equals).findFirst()
        .orElseThrow(() -> new UnsupportedOperationException("No such item available for this pool type: " + partClass));
    //noinspection unchecked
    return parts.stream()
        .map(part -> realPartClass.isAssignableFrom(part.getClass()) ? (R) part : null)
        .filter(Objects::nonNull).findAny().orElseGet(() -> {
          try {
            Constructor<? extends R> constructor = realPartClass.getDeclaredConstructor();
            constructor.setAccessible(true);
            ch.qos.logback.classic.Logger logger = loggerCtxt.getLogger(realPartClass);
            logger.addAppender(archiveAppender);
            R instance = constructor.newInstance();
            {
              Field ownerField = DataCacheItem.Stub.class.getDeclaredField("owner");
              ownerField.setAccessible(true);
              ownerField.set(instance, this);
            }
            {
              Field loggerField = DataCacheItem.Stub.class.getDeclaredField("logger");
              loggerField.setAccessible(true);
              loggerField.set(instance, logger);
            }

            parts.add(instance);
            return instance;
          } catch (InstantiationException | IllegalAccessException | NoSuchMethodException | InvocationTargetException | NoSuchFieldException e) {
            throw new RuntimeException(e);
          }
        });
  }

  public void flushConfig() {
    try(OutputStream stream = Files.newOutputStream(getConfigPath(), StandardOpenOption.CREATE)) {
      mapper.writeValue(stream, config);
    } catch (IOException | RuntimeException ignored) {
      log.warn("Unable to update log.txt entry");
    }
  }

  @NotNull
  private TPath getConfigPath() {
    return new TPath(getPath() + "/config.json");
  }

  public void propertyRead(String name) {
    if (!dependencyPool.empty()) {
      dependencyPool.peek().add(name);
    }
  }

  static {
    TConfig.get().setArchiveDetector(new TArchiveDetector("jool", new ZipDriver(IOPoolLocator.SINGLETON)));
  }

  private class ArchiveAppender extends AppenderBase<ILoggingEvent> {
    private final StdDateFormat dateFormat = new StdDateFormat();
    private final List<CharSequence> log = new ArrayList<>();

    @Override
    protected void append(ILoggingEvent eventObject) {
      if (dependencyPool.empty()) {
        throw new RuntimeException("The logger is not intended to be used outside cache items updates");
      }
      final String msg = dateFormat.format(new Date(eventObject.getTimeStamp())) + "\t" + eventObject.getLoggerName() + "\t" + eventObject.toString();
      log.add(msg);
      DataCacheImpl.log.info(msg);
    }

    Stream<CharSequence> flush() {
      return log.stream().onClose(log::clear);
    }
  }
}
