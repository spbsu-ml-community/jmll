package com.expleague.crawl.data.impl.system;

import com.expleague.crawl.data.Command;
import com.expleague.crawl.data.Message;

import java.util.concurrent.BlockingQueue;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
public class PingCommand implements Command {
  @Override
  public void execute(BlockingQueue<Message> responseQueue) {
    responseQueue.add(new PongMessage());
  }
}
