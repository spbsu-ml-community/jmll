package com.spbsu.ml.models.gpf.weblogmodel;

import com.spbsu.ml.models.gpf.Session;

/**
 * Created by irlab on 03.10.2014.
 */
public class BlockV1 extends Session.Block {
  public final ResultType resultType;
  public final ResultGrade resultGrade;

  public BlockV1(final Session.BlockType blockType, final int position) {
    this(blockType, null, position, null);
  }

  public BlockV1(final Session.BlockType blockType, final ResultType resultType,
                 final int position, final ResultGrade resultGrade)
  {
    super(blockType, position);
    this.resultType = resultType;
    this.resultGrade = resultGrade;
  }

  @Override
  public String toString() {
    return "Block{" + blockType +
           ", " + resultType +
           ", " + resultGrade +
           ", position=" + position +
           '}';
  }

  public enum ResultType {
    WEB, NEWS, IMAGES, DIRECT, VIDEO, OTHER
  }

  public enum ResultGrade {
    VITAL           (0.61), //0.69),
    USEFUL          (0.41), //0.47),
    RELEVANT_PLUS   (0.14), //0.45),
    RELEVANT_MINUS  (0.07), //0.44),
    IRRELEVANT      (0.03), //0.24),
    NOT_ASED        (0.10); //0.39);

    private final double pfound_value;
    //private final double ctr1_value;
    ResultGrade(final double pfound_value) {
      this.pfound_value = pfound_value;
    }

    public double getPfound_value() {
      return pfound_value;
    }
  }
}
