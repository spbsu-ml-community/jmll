package com.spbsu.exp.mus.classf;

import org.jetbrains.annotations.NotNull;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

import static java.text.MessageFormat.format;

/**
 * jmll
 * ksen
 * 11.April.2015 at 00:10
 */
public class Preprocessing {

  private static final Logger LOG = LoggerFactory.getLogger(Preprocessing.class);

  private final File dataStorage;
  private final File processedStorage;
  private final File preparedStorage;

  public Preprocessing(
      final @NotNull File dataStorage,
      final @NotNull File processedStorage,
      final @NotNull File preparedStorage
  ) {
    this.dataStorage = dataStorage;
    this.processedStorage = processedStorage;
    this.preparedStorage = preparedStorage;


    log(
        "Directory with inputs ''{0}''\n" +
            "Directory with outputs ''{1}''\n" +
            "Directory with prepared data ''{2}''",
        dataStorage,
        processedStorage,
        preparedStorage
    );
  }

  public void process() {
    final File[] genresDirectories = dataStorage.listFiles();

    mapDirectories(genresDirectories);

    try {
      traversal(genresDirectories);
      for (int i = 0; i < 20; i++) {
        queue.put(poison);
        forkJoinPool.execute(new Worker());
      }
      forkJoinPool.shutdown();
      forkJoinPool.awaitTermination(1000, TimeUnit.SECONDS);
    }
    catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  private void mapDirectories(final File[] directories) {
    for (int i = 0; i < directories.length; i++) {
      final String name = directories[i].getName();
      final File mappedSpectro = new File(processedStorage, name);
      final File mappedLR = new File(preparedStorage, name);

      log(
          "Mapped directory was created ''{0}'' -> ''{1}'', ''{2}''", name, mappedSpectro.mkdir(), mappedLR.mkdir()
      );
    }
  }

  private void traversal(final File[] directories) throws InterruptedException {
    for (final File directory : directories) {
      for (final File file : directory.listFiles()) {
        queue.put(file);
      }
    }
  }

  private final File poison = new File("POISON");
  private final ForkJoinPool forkJoinPool = new ForkJoinPool(20);
  private final BlockingQueue<File> queue = new LinkedBlockingQueue<>();

  private class Worker implements Runnable {
    @Override
    public void run() {
      try {
        File file;
        while (!(file = queue.take()).equals(poison)){
          final String inputName = file.getName();
          final String outputName = inputName.substring(0, inputName.lastIndexOf('.'));
          final String directoryName = file.getParentFile().getName();
          final File output = new File(processedStorage, directoryName + File.separator + outputName + ".png");

          final String command = "sox " + file.getAbsolutePath() +
              " -n spectrogram -lrh -o " + output.getAbsolutePath()
          ;
          final Process sox = Runtime.getRuntime().exec(command);
          log("{0} -> {1}", command, sox.waitFor());

          final BufferedImage spectro = ImageIO.read(output);

          final BufferedImage lStream = spectro.getSubimage(0, 1, 797, 256);
          final BufferedImage rStream = spectro.getSubimage(0, 258, 797, 256);

          ImageIO.write(lStream, "png", new File(preparedStorage, directoryName + File.separator + outputName + "-l"));
          ImageIO.write(rStream, "png", new File(preparedStorage, directoryName + File.separator + outputName + "-r"));
        }
      }
      catch (InterruptedException | IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  private void log(final String message, final Object ... objects) {
    LOG.info(format(message, objects));
  }

}
