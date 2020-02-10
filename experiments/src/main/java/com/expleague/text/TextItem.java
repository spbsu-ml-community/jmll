package com.expleague.text;

import com.expleague.commons.seq.CharSeq;
import com.expleague.ml.meta.DSItem;
import com.fasterxml.jackson.annotation.JsonProperty;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.adapters.XmlAdapter;
import javax.xml.bind.annotation.adapters.XmlJavaTypeAdapter;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

@XmlRootElement
public class TextItem implements DSItem {
  @XmlAttribute
  @XmlJavaTypeAdapter(CharSeqXmlAdapter.class)
  public CharSeq text;

  public TextItem(CharSeq text) {
    this.text = text;
  }

  @SuppressWarnings("unused")
  public TextItem() {}

  @Override
  public String id() {
    return Arrays.toString(md5.digest(text.toString().getBytes(StandardCharsets.UTF_8)));
  }

  public CharSeq text() {
    return text;
  }

  static final MessageDigest md5;
  static {
    try {
      md5 = MessageDigest.getInstance("MD5");
    }
    catch (NoSuchAlgorithmException e) {
      throw new RuntimeException(e);
    }
  }

  public static class CharSeqXmlAdapter extends XmlAdapter<String, CharSeq> {
    public CharSeq unmarshal(String var1) {
      return CharSeq.create(var1);
    }

    public String marshal(CharSeq var1) {
      return var1.toString();
    }
  }
}
