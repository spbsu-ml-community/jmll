package com.expleague.ml.methods.embedding;

import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.LongSeq;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.EmbeddingBuilderBase;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class CoocBuildTest {
  @Test
  public void testSymmetric() throws IOException {
    EmbeddingBuilderBasePublicMorozov morozov = new EmbeddingBuilderBasePublicMorozov();
    Path tempFile = Files.createTempFile("embedding", "test.txt");
    Files.write(tempFile, Arrays.asList(
        "Жили у бабуси",
        "Два весёлых гуся,",
        "Один - серый, другой - белый,",
        "Два весёлых гуся.",
        "Один - серый, другой - белый,",
        "Два весёлых гуся!",
        "Мыли гуси лапки",
        "В луже у канавки,",
        "Один - серый, другой - белый,",
        "Спрятались в канавке.",
        "Один - серый, другой - белый,",
        "Спрятались в канавке!",
        "Вот кричит бабуся:",
        "\"Ой, пропали гуси!",
        "Один - серый, другой - белый,",
        "Гуси мои, гуси!",
        "Один - серый, другой - белый,",
        "Гуси мои, гуси!\"",
        "Выходили гуси,",
        "Кланялись бабусе,",
        "Один - серый, другой - белый,",
        "Кланялись бабусе.",
        "Один - серый, другой - белый,",
        "Кланялись бабусе!")
    );
    morozov.window(Embedding.WindowType.FIXED, 2, 2).minWordCount(1).file(tempFile).build();

    Assert.assertEquals(27, morozov.dict().size());
    final LongSeq grayCooc = morozov.cooc(morozov.index("серый"));
    Assert.assertEquals(3, grayCooc.length());
    Assert.assertEquals("один:8.0 другой:8.0 белый:8.0", coocToString(morozov, grayCooc));
    final LongSeq gusiCooc = morozov.cooc(morozov.index("гуси"));
    Assert.assertEquals("гуси:4.0 мои:4.0 выходили:1.0 пропали:1.0 мыли:1.0 ой:1.0 лапки:1.0", coocToString(morozov, gusiCooc));
    Files.delete(tempFile);
  }

  private String coocToString(EmbeddingBuilderBasePublicMorozov morozov, LongSeq grayCooc) {
    return grayCooc.stream().mapToObj(l -> morozov.dict().get((int) (l >> 32)) + ":" + Float.intBitsToFloat((int) (l & 0xFFFFFFFFL))).collect(Collectors.joining(" "));
  }

  private static class EmbeddingBuilderBasePublicMorozov extends EmbeddingBuilderBase {
    @Override
    protected Embedding<CharSeq> fit() {
      return null;
    }

    @Override
    protected boolean isCoocNecessery() {
      return true;
    }

    @Override
    public LongSeq cooc(int i) {
      return super.cooc(i);
    }

    @Override
    public List<CharSeq> dict() {
      return super.dict();
    }

    @Override
    protected int index(CharSequence word) {
      return super.index(word);
    }
  }
}
